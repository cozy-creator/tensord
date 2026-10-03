//! NVML through `libnvidia-ml.so.1`, opened at runtime; no CUDA. Read only on each GPU's own
//! sampler thread, never on the API or journal loop: a driver stall stalls that thread alone.
use super::policy::Sample;
use std::{
    collections::BTreeMap,
    ffi::{c_char, c_int, c_uint, c_void, CString},
    io,
    sync::{Arc, Condvar, Mutex},
    time::Duration,
};

type Handle = *mut c_void;
type Init = unsafe extern "C" fn() -> c_int;
type ByIndex = unsafe extern "C" fn(c_uint, *mut Handle) -> c_int;
type ByUuid = unsafe extern "C" fn(*const c_char, *mut Handle) -> c_int;
type MemoryInfo = unsafe extern "C" fn(Handle, *mut Memory) -> c_int;
type Display = unsafe extern "C" fn(Handle, *mut c_int) -> c_int;
type Processes = unsafe extern "C" fn(Handle, *mut c_uint, *mut ProcessInfo) -> c_int;
type Uuid = unsafe extern "C" fn(Handle, *mut c_char, c_uint) -> c_int;
type DriverVersion = unsafe extern "C" fn(*mut c_char, c_uint) -> c_int;

const SUCCESS: c_int = 0;
const INSUFFICIENT_SIZE: c_int = 7;
/// `NVML_VALUE_NOT_AVAILABLE`: the driver does not account this process.
const NOT_AVAILABLE: u64 = u64::MAX;

#[repr(C)]
struct Memory {
    total: u64,
    free: u64,
    used: u64,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ProcessInfo {
    pid: c_uint,
    used: u64,
    gpu_instance: c_uint,
    compute_instance: c_uint,
}

/// One GPU's handle in an initialized NVML.
pub struct Device {
    /// `<GPU UUID>/<driver version>`: what a learned context is valid for.
    pub key: String,
    handle: Handle,
    memory: MemoryInfo,
    /// `nvmlDeviceGetDisplayActive`: a desktop is initialized on it. A connector with a
    /// display attached (`DisplayMode`, a dummy plug on many datacenter pods) is not one.
    display: Display,
    processes: Processes,
}
// SAFETY: NVML handles are process-global and its calls are thread-safe; each Device is read
// by its one sampler thread.
unsafe impl Send for Device {}

fn symbol<T>(library: *mut c_void, name: &str) -> io::Result<T> {
    let name = CString::new(name).map_err(io::Error::other)?;
    // SAFETY: dlsym on a library this process opened; the caller names the C signature.
    let found = unsafe { libc::dlsym(library, name.as_ptr()) };
    if found.is_null() {
        return Err(io::Error::other(format!("NVML lacks {name:?}")));
    }
    // SAFETY: a function pointer of the documented NVML signature `T`.
    Ok(unsafe { std::mem::transmute_copy::<*mut c_void, T>(&found) })
}

impl Device {
    /// `entry` as `CUDA_VISIBLE_DEVICES` spells it: an index, or a `GPU-…` UUID.
    pub fn open(entry: &str) -> io::Result<Self> {
        let path = CString::new("libnvidia-ml.so.1").map_err(io::Error::other)?;
        // SAFETY: dlopen of the driver's management library; it stays loaded for the process.
        let library = unsafe { libc::dlopen(path.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
        if library.is_null() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "libnvidia-ml.so.1 absent",
            ));
        }
        let init: Init = symbol(library, "nvmlInit_v2")?;
        // SAFETY: NVML functions with their documented signatures.
        unsafe {
            if init() != SUCCESS {
                return Err(io::Error::other("nvmlInit_v2 refused"));
            }
            let mut handle: Handle = std::ptr::null_mut();
            let found = match entry.parse::<c_uint>() {
                Ok(index) => {
                    symbol::<ByIndex>(library, "nvmlDeviceGetHandleByIndex_v2")?(index, &mut handle)
                }
                Err(_) => {
                    let uuid = CString::new(entry).map_err(io::Error::other)?;
                    symbol::<ByUuid>(library, "nvmlDeviceGetHandleByUUID")?(
                        uuid.as_ptr(),
                        &mut handle,
                    )
                }
            };
            if found != SUCCESS {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("NVML has no device {entry:?} ({found})"),
                ));
            }
            let text = |fill: &dyn Fn(*mut c_char) -> c_int| {
                let mut buffer = [0 as c_char; 96];
                (fill(buffer.as_mut_ptr()) == SUCCESS).then(|| {
                    std::ffi::CStr::from_ptr(buffer.as_ptr())
                        .to_string_lossy()
                        .into_owned()
                })
            };
            let uuid: Uuid = symbol(library, "nvmlDeviceGetUUID")?;
            let driver: DriverVersion = symbol(library, "nvmlSystemGetDriverVersion")?;
            let key = format!(
                "{}/{}",
                text(&|b| uuid(handle, b, 96)).unwrap_or_else(|| entry.into()),
                text(&|b| driver(b, 96)).unwrap_or_default()
            );
            Ok(Self {
                key,
                handle,
                memory: symbol(library, "nvmlDeviceGetMemoryInfo")?,
                display: symbol(library, "nvmlDeviceGetDisplayActive")?,
                processes: symbol(library, "nvmlDeviceGetComputeRunningProcesses_v3")?,
            })
        }
    }

    pub fn sample(&self) -> io::Result<Sample> {
        let mut memory = Memory {
            total: 0,
            free: 0,
            used: 0,
        };
        // SAFETY: documented NVML calls on a live handle with owned out-parameters.
        unsafe {
            if (self.memory)(self.handle, &mut memory) != SUCCESS {
                return Err(io::Error::other("nvmlDeviceGetMemoryInfo failed"));
            }
            let mut state: c_int = 0;
            let display = (self.display)(self.handle, &mut state) == SUCCESS && state == 1;
            Ok(Sample {
                total: memory.total,
                free: memory.free,
                display,
                processes: self.process_table(),
            })
        }
    }

    /// pid -> bytes; None when the driver cannot list or account every process.
    unsafe fn process_table(&self) -> Option<BTreeMap<u32, u64>> {
        let mut count: c_uint = 0;
        let mut rows = Vec::new();
        loop {
            rows.resize(count as usize, ProcessInfo::default());
            let answer = (self.processes)(self.handle, &mut count, rows.as_mut_ptr());
            match answer {
                SUCCESS => break,
                INSUFFICIENT_SIZE => count += 4,
                _ => return None,
            }
        }
        rows.truncate(count as usize);
        rows.iter()
            .map(|row| (row.used != NOT_AVAILABLE).then_some((row.pid, row.used)))
            .collect()
    }
}

struct State {
    latest: Option<Sample>,
    asked: u64,
    answered: u64,
    stopped: bool,
}

/// One GPU's sampler thread: a reading every second, and one on demand before a decision.
pub struct Sampler {
    shared: Arc<(Mutex<State>, Condvar)>,
}

impl Sampler {
    /// `observe` runs on the sampler thread after every reading (the floor watchdog).
    pub fn start(device: Device, observe: impl Fn(&Sample) + Send + 'static) -> io::Result<Self> {
        let shared = Arc::new((
            Mutex::new(State {
                latest: None,
                asked: 0,
                answered: 0,
                stopped: false,
            }),
            Condvar::new(),
        ));
        let thread = shared.clone();
        std::thread::Builder::new()
            .name("gpu-memory-sampler".into())
            .spawn(move || loop {
                let reading = device.sample().ok();
                let (lock, changed) = &*thread;
                let mut state = lock.lock().unwrap();
                state.latest = reading.clone();
                state.answered = state.asked;
                changed.notify_all();
                if state.stopped {
                    return;
                }
                drop(state);
                if let Some(sample) = &reading {
                    observe(sample);
                }
                let state = lock.lock().unwrap();
                let _ = changed
                    .wait_timeout_while(state, Duration::from_secs(1), |s| {
                        s.asked == s.answered && !s.stopped
                    })
                    .unwrap();
            })?;
        Ok(Self { shared })
    }

    /// A reading taken after this call: waits for the sampler thread, not for a clock.
    pub fn now(&self) -> Option<Sample> {
        let (lock, changed) = &*self.shared;
        let mut state = lock.lock().unwrap();
        state.asked += 1;
        let ticket = state.asked;
        changed.notify_all();
        let state = changed
            .wait_while(state, |s| s.answered < ticket && !s.stopped)
            .unwrap();
        state.latest.clone()
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        let (lock, changed) = &*self.shared;
        lock.lock().unwrap().stopped = true;
        changed.notify_all();
    }
}
