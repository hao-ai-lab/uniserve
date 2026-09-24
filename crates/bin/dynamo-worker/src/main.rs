// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Entry point of `uniserve-dynamo-worker`.
//!
//! `main` is synchronous: Dynamo's `run_raw` builds its own async runtime and
//! drives the engine through start, registration, request serving and
//! cleanup.

use std::sync::Arc;

use anyhow::Result;

fn main() -> Result<()> {
    let (engine, config) = uniserve_dynamo_worker::DynamoFastH3Engine::from_args()?;
    dynamo_backend_common::run_raw(Arc::new(engine), config)
}
