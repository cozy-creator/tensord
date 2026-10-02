//! Typed descriptor-source control exchange shared by GPU pilot and CPU qualification.
use crate::{
    device_executor::{self, Answer, Frame, Kind},
    model_sources::{ModelSources, SourceRequest, SourceRole},
};
use std::{fs::File, io};

/// One shared typed source exchange for a negotiated SDK provider or CPU diagnostic consumer.
pub fn answer(sources: &mut ModelSources, frame: &Frame) -> io::Result<(Answer, File)> {
    if frame.kind != Kind::ModelSourceRead {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "not a model source request",
        ));
    }
    let role = match frame.role {
        device_executor::SourceRole::Header => SourceRole::Header,
        device_executor::SourceRole::Asset => SourceRole::Asset,
        device_executor::SourceRole::Object => SourceRole::Object,
        device_executor::SourceRole::Unknown => SourceRole::Unknown,
    };
    let grant = sources.read(&SourceRequest {
        manifest: frame.manifest.clone(),
        role,
        name: frame.name.clone(),
        length: frame.length,
    })?;
    let mut answer = Answer::unavailable(frame.seq);
    answer.ok = true;
    answer.code.clear();
    answer.detail.clear();
    answer.descriptor = true;
    answer.sha256 = grant.sha256;
    answer.length = grant.length;
    Ok((answer, grant.file))
}
