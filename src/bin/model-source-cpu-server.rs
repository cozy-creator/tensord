//! CPU diagnostic owner for the exact device pilot's model source exchange.
use cozy_machine::{
    device_executor::{read_frame, write_answer},
    model_source_driver,
    model_sources::{ModelSources, SelectedManifest},
    protocol,
};
use std::{
    fs, io,
    os::unix::{fs::PermissionsExt, net::UnixListener},
    path::PathBuf,
};

fn main() -> io::Result<()> {
    let mut args = std::env::args().skip(1);
    let store = PathBuf::from(
        args.next()
            .ok_or_else(|| io::Error::other("store required"))?,
    );
    let manifest = args
        .next()
        .ok_or_else(|| io::Error::other("manifest required"))?;
    let components = args
        .next()
        .ok_or_else(|| io::Error::other("components required"))?
        .split(',')
        .map(str::to_owned)
        .collect();
    let socket = PathBuf::from(
        args.next()
            .ok_or_else(|| io::Error::other("socket required"))?,
    );
    if args.next().is_some() {
        return Err(io::Error::other("unexpected argument"));
    }
    let mut sources = ModelSources::open(
        &store,
        &[SelectedManifest {
            manifest,
            components,
        }],
    )?;
    let listener = UnixListener::bind(&socket)?;
    fs::set_permissions(&socket, fs::Permissions::from_mode(0o600))?;
    println!("ready {}", socket.display());
    let (mut stream, _) = listener.accept()?;
    let mut exports = 0u64;
    while let Some(frame) = read_frame(&mut stream)? {
        let (answer, file) = model_source_driver::answer(&mut sources, &frame)?;
        write_answer(&mut stream, &answer)?;
        protocol::send_fd(&stream, &file)?;
        drop(file);
        exports += 1;
    }
    println!("exports={exports} receiver_closed=true");
    Ok(())
}
