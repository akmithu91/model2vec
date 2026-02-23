//! JNI bridge — Linux x86_64.
//!
//! # Classes exposed
//!
//! ## Model2Vec
//!   - Constructor stores max_length + batch_size alongside Arc<StaticModel>
//!   - encode(String[]) → float[][] via set_float_array_region (bulk memcpy per row)
//!
//! ## Chunker
//!   - chunk(String) → int[] of flat byte-offset pairs [s0,e0, s1,e1, ...]
//!   - Single set_int_array_region call — zero new_string() crossings
//!   - Java reconstructs chunks: new String(text.getBytes(UTF_8), s, e-s, UTF_8)
//!
//! # Key design choices
//!   - Primitive JNI array ops only — no ArrayList, no reflective call_method
//!   - All arrays pre-sized before any JVM allocation
//!   - Arc<StaticModel> — Sync, no mutex, fully concurrent encode calls
//!   - Box<T> as jlong nativeHandle — instance-based, no global state

mod chunker;

use chunker::{Chunker, SnapCfg};
use jni::objects::{JObject, JObjectArray, JString, JFloatArray, JIntArray};
use jni::sys::{jboolean, jfloat, jint, jlong, jobjectArray, JNI_TRUE};
use jni::JNIEnv;
use model2vec_rs::model::StaticModel;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Model wrapper
// ---------------------------------------------------------------------------

struct ModelHandle {
    model:      Arc<StaticModel>,
    max_length: Option<usize>,
    batch_size: usize,
}

// ---------------------------------------------------------------------------
// Throw helpers — explicit typed returns, no generic Default trick.
// *mut _jobject does not implement Default so the generic approach fails.
// ---------------------------------------------------------------------------

fn throw_obj(env: &mut JNIEnv, msg: &str) -> jobjectArray {
    let _ = env.throw_new("java/lang/RuntimeException", msg);
    std::ptr::null_mut()
}

fn throw_ints(env: &mut JNIEnv, msg: &str) -> jni::sys::jintArray {
    let _ = env.throw_new("java/lang/RuntimeException", msg);
    std::ptr::null_mut()
}

fn throw_void(env: &mut JNIEnv, msg: &str) {
    let _ = env.throw_new("java/lang/RuntimeException", msg);
}

// ---------------------------------------------------------------------------
// Handle field accessors
// ---------------------------------------------------------------------------

fn get_handle(env: &mut JNIEnv, this: &JObject) -> Result<jlong, jni::errors::Error> {
    env.get_field(this, "nativeHandle", "J")?.j()
}

fn set_handle(env: &mut JNIEnv, this: &JObject, h: jlong) -> Result<(), jni::errors::Error> {
    env.set_field(this, "nativeHandle", "J", h.into())
}

// ---------------------------------------------------------------------------
// Array helpers
// ---------------------------------------------------------------------------

fn read_string_array(env: &mut JNIEnv, arr: &JObjectArray) -> Result<Vec<String>, String> {
    let len = env.get_array_length(arr)
        .map_err(|e| format!("get_array_length: {e}"))?;
    let mut out = Vec::with_capacity(len as usize);
    for i in 0..len {
        let elem = env.get_object_array_element(arr, i)
            .map_err(|e| format!("get_object_array_element({i}): {e}"))?;
        let s: String = env.get_string(&JString::from(elem))
            .map_err(|e| format!("get_string({i}): {e}"))?.into();
        out.push(s);
    }
    Ok(out)
}

/// Build float[][] — one set_float_array_region (bulk memcpy) per row.
/// Returns raw jobjectArray so the borrow on env ends before the call site
/// needs env again (e.g. for throw_obj on the Err path).
fn make_float_2d(env: &mut JNIEnv, data: Vec<Vec<f32>>) -> Result<jobjectArray, String> {
    let cls   = env.find_class("[F")
        .map_err(|e| format!("find_class [F: {e}"))?;
    let outer = env.new_object_array(data.len() as jint, &cls, &JObject::null())
        .map_err(|e| format!("new_object_array outer: {e}"))?;
    for (i, row) in data.iter().enumerate() {
        let inner: JFloatArray = env.new_float_array(row.len() as jint)
            .map_err(|e| format!("new_float_array({i}): {e}"))?;
        env.set_float_array_region(&inner, 0, row.as_slice() as &[jfloat])
            .map_err(|e| format!("set_float_array_region({i}): {e}"))?;
        env.set_object_array_element(&outer, i as jint, &inner)
            .map_err(|e| format!("set_object_array_element({i}): {e}"))?;
    }
    // into_raw() here ends the borrow on env before we return to the call site
    Ok(outer.into_raw())
}

// ===========================================================================
// Model2Vec
// ===========================================================================

#[no_mangle]
pub extern "system" fn Java_com_model2vec_Model2Vec_nativeCreate<'local>(
    mut env:    JNIEnv<'local>,
    this:       JObject<'local>,
    model_id:   JString<'local>,
    normalize:  jboolean,
    max_length: jint,
    batch_size: jint,
) {
    let id: String = match env.get_string(&model_id) {
        Ok(s)  => s.into(),
        Err(e) => return throw_void(&mut env, &format!("read modelId: {e}")),
    };
    let model = match StaticModel::from_pretrained(&id, None, Some(normalize == JNI_TRUE), None) {
        Ok(m)  => m,
        Err(e) => return throw_void(&mut env, &format!("load model '{id}': {e}")),
    };
    let mh = ModelHandle {
        model:      Arc::new(model),
        max_length: if max_length > 0 { Some(max_length as usize) } else { None },
        batch_size: if batch_size > 0 { batch_size as usize } else { 1024 },
    };
    let handle = Box::into_raw(Box::new(mh)) as jlong;
    if let Err(e) = set_handle(&mut env, &this, handle) {
        unsafe { drop(Box::from_raw(handle as *mut ModelHandle)); }
        throw_void(&mut env, &format!("store nativeHandle: {e}"));
    }
}

#[no_mangle]
pub extern "system" fn Java_com_model2vec_Model2Vec_nativeDestroy<'local>(
    mut env: JNIEnv<'local>,
    this:    JObject<'local>,
) {
    match get_handle(&mut env, &this) {
        Ok(0)  => {}
        Ok(h)  => {
            unsafe { drop(Box::from_raw(h as *mut ModelHandle)); }
            let _ = set_handle(&mut env, &this, 0);
        }
        Err(e) => throw_void(&mut env, &format!("get nativeHandle: {e}")),
    }
}

#[no_mangle]
pub extern "system" fn Java_com_model2vec_Model2Vec_nativeEncode<'local>(
    mut env:   JNIEnv<'local>,
    this:      JObject<'local>,
    sentences: JObjectArray<'local>,
) -> jobjectArray {
    let handle = match get_handle(&mut env, &this) {
        Ok(0)  => return throw_obj(&mut env, "Model already closed"),
        Ok(h)  => h,
        Err(e) => return throw_obj(&mut env, &format!("get nativeHandle: {e}")),
    };
    let mh: &ModelHandle = unsafe { &*(handle as *const ModelHandle) };

    let texts = match read_string_array(&mut env, &sentences) {
        Ok(v)  => v,
        Err(e) => return throw_obj(&mut env, &e),
    };

    let embeddings = mh.model.encode_with_args(&texts, mh.max_length, mh.batch_size);

    match make_float_2d(&mut env, embeddings) {
        Ok(raw) => raw,
        Err(e)  => throw_obj(&mut env, &e),
    }
}

// ===========================================================================
// Chunker
// ===========================================================================

#[no_mangle]
pub extern "system" fn Java_com_model2vec_Chunker_nativeCreate<'local>(
    mut env:            JNIEnv<'local>,
    this:               JObject<'local>,
    repo_or_path:       JString<'local>,
    chunk_size:         jint,
    stride:             jint,
    max_forward_bytes:  jint,
    max_backward_bytes: jint,
    prefer_paragraph:   jboolean,
    trim_whitespace:    jboolean,
) {
    let path: String = match env.get_string(&repo_or_path) {
        Ok(s)  => s.into(),
        Err(e) => return throw_void(&mut env, &format!("read tokenizerRepoOrPath: {e}")),
    };
    let snap_cfg = SnapCfg {
        max_forward_bytes:  max_forward_bytes  as usize,
        max_backward_bytes: max_backward_bytes as usize,
        prefer_paragraph:   prefer_paragraph   == JNI_TRUE,
        trim_whitespace:    trim_whitespace     == JNI_TRUE,
    };
    let c = match Chunker::new(&path, chunk_size as usize, stride as usize, snap_cfg) {
        Ok(c)  => c,
        Err(e) => return throw_void(&mut env, &format!("Chunker::new: {e}")),
    };
    let handle = Box::into_raw(Box::new(c)) as jlong;
    if let Err(e) = set_handle(&mut env, &this, handle) {
        unsafe { drop(Box::from_raw(handle as *mut Chunker)); }
        throw_void(&mut env, &format!("store nativeHandle: {e}"));
    }
}

#[no_mangle]
pub extern "system" fn Java_com_model2vec_Chunker_nativeDestroy<'local>(
    mut env: JNIEnv<'local>,
    this:    JObject<'local>,
) {
    match get_handle(&mut env, &this) {
        Ok(0)  => {}
        Ok(h)  => {
            unsafe { drop(Box::from_raw(h as *mut Chunker)); }
            let _ = set_handle(&mut env, &this, 0);
        }
        Err(e) => throw_void(&mut env, &format!("get nativeHandle: {e}")),
    }
}

/// Returns flat int[] of byte-offset pairs [start0, end0, start1, end1, ...].
/// Single set_int_array_region — one bulk memcpy, zero new_string() crossings.
#[no_mangle]
pub extern "system" fn Java_com_model2vec_Chunker_nativeChunk<'local>(
    mut env: JNIEnv<'local>,
    this:    JObject<'local>,
    text:    JString<'local>,
) -> jni::sys::jintArray {
    let handle = match get_handle(&mut env, &this) {
        Ok(0)  => return throw_ints(&mut env, "Chunker already closed"),
        Ok(h)  => h,
        Err(e) => return throw_ints(&mut env, &format!("get nativeHandle: {e}")),
    };
    let chunker: &Chunker = unsafe { &*(handle as *const Chunker) };

    let rust_text: String = match env.get_string(&text) {
        Ok(s)  => s.into(),
        Err(e) => return throw_ints(&mut env, &format!("read text: {e}")),
    };

    let offsets: Vec<i32> = match chunker.chunk(&rust_text) {
        Ok(v)  => v,
        Err(e) => return throw_ints(&mut env, &format!("chunk: {e}")),
    };

    let arr: JIntArray = match env.new_int_array(offsets.len() as jint) {
        Ok(a)  => a,
        Err(e) => return throw_ints(&mut env, &format!("new_int_array: {e}")),
    };
    if let Err(e) = env.set_int_array_region(&arr, 0, &offsets) {
        return throw_ints(&mut env, &format!("set_int_array_region: {e}"));
    }

    arr.into_raw()
}
