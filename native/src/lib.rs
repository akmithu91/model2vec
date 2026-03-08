use chunker::{Chunker, SnapCfg};
use model2vec_rs::model::StaticModel;
use std::sync::Arc;
use std::ffi::CStr;
use std::os::raw::c_char;
use std::slice;

mod chunker;

// ---------------------------------------------------------------------------
// Model wrapper
// ---------------------------------------------------------------------------

pub struct ModelHandle {
    model:      Arc<StaticModel>,
    max_length: Option<usize>,
    batch_size: usize,
}

#[no_mangle]
pub extern "C" fn model2vec_create(
    model_id: *const c_char,
    normalize: bool,
    max_length: i32,
    batch_size: i32,
) -> *mut ModelHandle {
    if model_id.is_null() { return std::ptr::null_mut(); }
    let c_str = unsafe { CStr::from_ptr(model_id) };
    let id = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };

    let model = match StaticModel::from_pretrained(id, None, Some(normalize), None) {
        Ok(m) => m,
        Err(_) => return std::ptr::null_mut(),
    };

    let mh = ModelHandle {
        model: Arc::new(model),
        max_length: if max_length > 0 { Some(max_length as usize) } else { None },
        batch_size: if batch_size > 0 { batch_size as usize } else { 1024 },
    };
    Box::into_raw(Box::new(mh))
}

#[no_mangle]
pub extern "C" fn model2vec_free(handle: *mut ModelHandle) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle)); }
    }
}

#[no_mangle]
pub extern "C" fn model2vec_get_dim(handle: *mut ModelHandle) -> i32 {
    if handle.is_null() { return -1; }
    let mh = unsafe { &*handle };
    // Encode a dummy string to find the dimension if there's no direct method
    let dummy = vec!["test".to_string()];
    let embeddings = mh.model.encode_with_args(&dummy, mh.max_length, 1);
    if embeddings.is_empty() { return -1; }
    embeddings[0].len() as i32
}

#[no_mangle]
pub extern "C" fn model2vec_encode(
    handle: *mut ModelHandle,
    sentences: *const *const c_char,
    sentence_count: usize,
    out_ptr: *mut f32,
) -> i32 {
    if handle.is_null() || sentences.is_null() || out_ptr.is_null() {
        return -1;
    }
    let mh = unsafe { &*handle };
    
    let mut texts = Vec::with_capacity(sentence_count);
    let s_ptrs = unsafe { slice::from_raw_parts(sentences, sentence_count) };
    for &ptr in s_ptrs {
        if ptr.is_null() { return -2; }
        let c_str = unsafe { CStr::from_ptr(ptr) };
        match c_str.to_str() {
            Ok(s) => texts.push(s.to_string()),
            Err(_) => return -3,
        }
    }

    let embeddings = mh.model.encode_with_args(&texts, mh.max_length, mh.batch_size);
    if embeddings.is_empty() { return 0; }
    
    let dim = embeddings[0].len();
    let out_slice = unsafe { slice::from_raw_parts_mut(out_ptr, sentence_count * dim) };
    
    for (i, row) in embeddings.into_iter().enumerate() {
        let start = i * dim;
        out_slice[start..start + dim].copy_from_slice(&row);
    }
    
    0
}

// ---------------------------------------------------------------------------
// Chunker
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn chunker_create(
    repo_or_path: *const c_char,
    chunk_size: i32,
    stride: i32,
    max_forward_bytes: i32,
    max_backward_bytes: i32,
    prefer_paragraph: bool,
    trim_whitespace: bool,
) -> *mut Chunker {
    if repo_or_path.is_null() { return std::ptr::null_mut(); }
    let c_str = unsafe { CStr::from_ptr(repo_or_path) };
    let path = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };
    
    let snap_cfg = SnapCfg {
        max_forward_bytes:  max_forward_bytes  as usize,
        max_backward_bytes: max_backward_bytes as usize,
        prefer_paragraph:   prefer_paragraph,
        trim_whitespace:    trim_whitespace,
    };
    
    let c = match Chunker::new(path, chunk_size as usize, stride as usize, snap_cfg) {
        Ok(c) => c,
        Err(_) => return std::ptr::null_mut(),
    };
    
    Box::into_raw(Box::new(c))
}

#[no_mangle]
pub extern "C" fn chunker_free(handle: *mut Chunker) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle)); }
    }
}

#[no_mangle]
pub extern "C" fn chunker_chunk(
    handle: *mut Chunker,
    text: *const c_char,
    out_len: *mut usize,
) -> *mut i32 {
    if handle.is_null() || text.is_null() || out_len.is_null() {
        return std::ptr::null_mut();
    }
    let chunker = unsafe { &*handle };
    let c_str = unsafe { CStr::from_ptr(text) };
    let rust_text = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };
    
    let offsets = match chunker.chunk(rust_text) {
        Ok(v) => v,
        Err(_) => return std::ptr::null_mut(),
    };
    
    unsafe { *out_len = offsets.len(); }
    let mut offsets = offsets.into_boxed_slice();
    let ptr = offsets.as_mut_ptr();
    std::mem::forget(offsets);
    ptr
}

#[no_mangle]
pub unsafe extern "C" fn chunker_free_offsets(ptr: *mut i32, len: usize) {
    if !ptr.is_null() {
        drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len)));
    }
}
