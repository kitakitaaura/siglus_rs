#![cfg(target_arch = "wasm32")]

//! Browser directory-backed virtual file access for the Siglus wasm port.
//!
//! The browser cannot open native paths from wasm. The JavaScript launcher must
//! let the user select the original game directory, build a path -> File object
//! index, and expose the synchronous functions imported below. Rust only passes
//! normalized Siglus relative paths such as `Gameexe.ini`, `Scene.pck`,
//! `bgm/BGM016.nwa`, or `g00/sys10_tt01.g00`.
//!
//! This module intentionally uses path-only requests. File ids, offsets, and
//! browser handles are private to the JavaScript side. It does not package the
//! game and does not read file contents during directory scanning.

use anyhow::{Context, Result, anyhow, bail};
use js_sys::{Array, Uint8Array};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_name = siglusFileExists)]
    fn siglus_file_exists(path: &str) -> bool;

    #[wasm_bindgen(js_name = siglusReadFile)]
    fn siglus_read_file(path: &str) -> Uint8Array;

    #[wasm_bindgen(js_name = siglusListDir)]
    fn siglus_list_dir(path: &str) -> Array;

    #[wasm_bindgen(js_name = siglusKnownFileCount)]
    fn siglus_known_file_count() -> u32;

    // noberu: ranged reads, sizes and writes. A 433 MB movie read whole (twice)
    // does not fit a 32-bit heap, and saves need somewhere to go.
    #[wasm_bindgen(js_name = siglusReadRange)]
    fn siglus_read_range(path: &str, offset: f64, len: u32) -> Uint8Array;

    #[wasm_bindgen(js_name = siglusFileSize)]
    fn siglus_file_size(path: &str) -> f64;

    #[wasm_bindgen(js_name = siglusWriteFile)]
    fn siglus_write_file(path: &str, data: &[u8]);
}

/// Path-based filesystem surface used by the wasm port.
///
/// The first wasm implementation deliberately supports whole-file reads only.
/// That matches the current Siglus resource model where `Scene.pck`, G00, NWA,
/// OMV and other resources are individual files. If a specific large resource
/// later proves too expensive as a whole-file read, streaming/range I/O should
/// be added behind this trait without changing engine-level resource semantics.
pub trait SiglusVfs {
    fn exists(&self, path: &str) -> bool;
    fn read_all(&self, path: &str) -> Result<Vec<u8>>;
    fn list_dir(&self, path: &str) -> Result<Vec<String>>;
    /// Up to `len` bytes from `offset`; shorter at the end of the file.
    fn read_range(&self, path: &str, offset: u64, len: usize) -> Result<Vec<u8>>;
    fn file_size(&self, path: &str) -> Result<u64>;
    /// Creates or replaces the file (saves). The host keeps it.
    fn write_all(&self, path: &str, data: &[u8]) -> Result<()>;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct WasmDirectoryVfs;

impl WasmDirectoryVfs {
    pub fn new() -> Self {
        Self
    }

    pub fn known_file_count(&self) -> u32 {
        siglus_known_file_count()
    }
}

impl SiglusVfs for WasmDirectoryVfs {
    fn exists(&self, path: &str) -> bool {
        siglus_file_exists(&normalize_siglus_path(path))
    }

    fn read_all(&self, path: &str) -> Result<Vec<u8>> {
        let normalized = normalize_siglus_path(path);
        if normalized.is_empty() {
            bail!("empty wasm resource path");
        }
        if !siglus_file_exists(&normalized) {
            bail!("wasm resource not found: {normalized}");
        }

        let arr = siglus_read_file(&normalized);
        let len = arr.length() as usize;
        let mut out = vec![0u8; len];
        arr.copy_to(&mut out);
        Ok(out)
    }

    fn read_range(&self, path: &str, offset: u64, len: usize) -> Result<Vec<u8>> {
        let normalized = normalize_siglus_path(path);
        if !siglus_file_exists(&normalized) {
            bail!("wasm resource not found: {normalized}");
        }
        let len = u32::try_from(len).map_err(|_| anyhow!("range too large: {len}"))?;
        let arr = siglus_read_range(&normalized, offset as f64, len);
        let mut out = vec![0u8; arr.length() as usize];
        arr.copy_to(&mut out);
        Ok(out)
    }

    fn file_size(&self, path: &str) -> Result<u64> {
        let normalized = normalize_siglus_path(path);
        if !siglus_file_exists(&normalized) {
            bail!("wasm resource not found: {normalized}");
        }
        Ok(siglus_file_size(&normalized).max(0.0) as u64)
    }

    fn write_all(&self, path: &str, data: &[u8]) -> Result<()> {
        let normalized = normalize_siglus_path(path);
        if normalized.is_empty() {
            bail!("empty wasm resource path");
        }
        siglus_write_file(&normalized, data);
        Ok(())
    }

    fn list_dir(&self, path: &str) -> Result<Vec<String>> {
        let normalized = normalize_siglus_path(path);
        let array = siglus_list_dir(&normalized);
        let mut out = Vec::with_capacity(array.length() as usize);
        for value in array.iter() {
            let Some(s) = value.as_string() else {
                return Err(anyhow!(
                    "siglusListDir returned a non-string entry for {normalized}"
                ));
            };
            out.push(s);
        }
        Ok(out)
    }
}

/// `Read + Seek` over one file through ranged reads, so a decoder can walk a
/// large container (a 433 MB OMV) without the whole thing in memory. Reads
/// go through a sliding window, so sequential Ogg reads cost one host call
/// per window rather than one per read.
pub struct VfsSeekReader {
    path: String,
    len: u64,
    pos: u64,
    window_start: u64,
    window: Vec<u8>,
}

const VFS_READER_WINDOW: usize = 256 * 1024;

impl VfsSeekReader {
    pub fn open(path: &str) -> Result<Self> {
        let path = normalize_siglus_path(path);
        let len = WasmDirectoryVfs::new().file_size(&path)?;
        Ok(Self { path, len, pos: 0, window_start: 0, window: Vec::new() })
    }

    pub fn len(&self) -> u64 {
        self.len
    }
}

impl std::io::Read for VfsSeekReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if buf.is_empty() || self.pos >= self.len {
            return Ok(0);
        }
        let in_window = self.pos >= self.window_start
            && self.pos < self.window_start + self.window.len() as u64;
        if !in_window {
            let want = VFS_READER_WINDOW.max(buf.len()).min((self.len - self.pos) as usize);
            self.window = WasmDirectoryVfs::new()
                .read_range(&self.path, self.pos, want)
                .map_err(|e| std::io::Error::other(format!("{e:#}")))?;
            self.window_start = self.pos;
            if self.window.is_empty() {
                return Ok(0);
            }
        }
        let at = (self.pos - self.window_start) as usize;
        let n = buf.len().min(self.window.len() - at);
        buf[..n].copy_from_slice(&self.window[at..at + n]);
        self.pos += n as u64;
        Ok(n)
    }
}

impl std::io::Seek for VfsSeekReader {
    fn seek(&mut self, to: std::io::SeekFrom) -> std::io::Result<u64> {
        let next = match to {
            std::io::SeekFrom::Start(n) => n as i128,
            std::io::SeekFrom::End(n) => self.len as i128 + n as i128,
            std::io::SeekFrom::Current(n) => self.pos as i128 + n as i128,
        };
        if next < 0 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "seek before start"));
        }
        self.pos = next as u64;
        Ok(self.pos)
    }
}

/// Convenience export for JavaScript-side diagnostics.
#[wasm_bindgen]
pub fn siglus_wasm_vfs_file_count() -> u32 {
    WasmDirectoryVfs::new().known_file_count()
}

/// Convenience export for JavaScript-side diagnostics.
#[wasm_bindgen]
pub fn siglus_wasm_vfs_exists(path: &str) -> bool {
    WasmDirectoryVfs::new().exists(path)
}

/// Convenience export for JavaScript-side diagnostics.
#[wasm_bindgen]
pub fn siglus_wasm_vfs_read_len(path: &str) -> std::result::Result<u32, JsValue> {
    let bytes = WasmDirectoryVfs::new()
        .read_all(path)
        .with_context(|| format!("read wasm resource {path}"))
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(bytes.len() as u32)
}

pub fn normalize_siglus_path(path: &str) -> String {
    path.replace('\\', "/")
        .split('/')
        .filter(|part| !part.is_empty() && *part != ".")
        .collect::<Vec<_>>()
        .join("/")
}
