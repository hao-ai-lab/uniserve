//! Startup storage bounds shared by model workers and their resource owners.

use crate::{Error, Result};

pub const DEFAULT_NUM_UNITS: u64 = 4096;

/// The physical unit pool that fits beside a rank's fixed allocations.
#[derive(Clone, Copy, Debug)]
pub struct KVCapacity {
    pub unit_bytes: u64,
    pub num_units: u64,
}

impl KVCapacity {
    /// Explicit token capacity counts whole pages of every cache group.
    /// Automatic sizing gives the pool the remaining complete units; both
    /// modes must leave room for the sentinel and one page of every group.
    #[allow(clippy::too_many_arguments)]
    pub fn derive(
        pages: &[(u32, u32)],
        tokens: Option<u64>,
        unit_bytes: u64,
        available_bytes: Option<u64>,
        floor: u64,
        default_units: u64,
        resident_copies: u64,
        co_resident_units: u64,
    ) -> Result<Self> {
        if pages.is_empty()
            || pages
                .iter()
                .any(|&(tokens, units)| tokens == 0 || units == 0)
            || unit_bytes == 0
            || floor == 0
            || resident_copies == 0
        {
            return Err(Error::Invalid(
                "KV capacity dimensions must be positive".into(),
            ));
        }

        let minimum = floor.max(
            1 + pages
                .iter()
                .map(|&(_, units)| u64::from(units))
                .sum::<u64>(),
        );
        let num_units = if let Some(tokens) = tokens {
            if tokens == 0 {
                return Err(Error::Invalid(
                    "configured KV token capacity must be positive".into(),
                ));
            }
            minimum.max(units_for_tokens(pages, tokens))
        } else if let Some(available) = available_bytes {
            let units =
                (available / unit_bytes).saturating_sub(co_resident_units) / resident_copies;
            if units < minimum {
                return Err(Error::Resource(
                    "device storage grant cannot hold the required KV pool",
                ));
            }
            units
        } else {
            minimum.max(default_units)
        };

        if available_bytes.is_some_and(|available| {
            (resident_copies * num_units + co_resident_units) * unit_bytes > available
        }) {
            return Err(Error::Resource(
                "configured KV storage exceeds the device storage grant",
            ));
        }

        Ok(Self {
            unit_bytes,
            num_units,
        })
    }
}

/// Count whole token pages across all cache groups, rounding down per group.
pub fn units_for_tokens(pages: &[(u32, u32)], tokens: u64) -> u64 {
    pages
        .iter()
        .map(|&(page, units)| tokens / u64::from(page) * u64::from(units))
        .sum()
}

/// Unresolved calls of a token worker, including a producer and its consumer
/// when a depth-one batch can contain both.
pub fn call_window(depth: usize, calls: usize) -> Result<usize> {
    if depth == 0 || calls == 0 {
        return Err(Error::Invalid(
            "call-window sizing requires positive bounds".into(),
        ));
    }

    Ok((depth * calls).min(depth.max(2)))
}

/// Media slots keep two unresolved outputs and one position for retirement.
pub fn request_tensor_window(depth: usize, slots: usize) -> Result<usize> {
    if slots == 0 || depth / slots < 3 {
        return Err(Error::Invalid(
            "request tensor pipeline requires two unresolved outputs per slot".into(),
        ));
    }

    Ok(depth / slots - 1)
}

/// Storage and concurrency bounds consumed by the worker's physical owners.
#[derive(Clone, Copy, Debug)]
pub struct ArenaCapacity {
    pub latent_pool_bytes: u64,
    pub tensor_store: usize,
    pub device_product_bytes: u64,
    pub transfer_bytes: u64,
    pub transfer_tickets: usize,
    pub host_lane_inflight: usize,
}

impl ArenaCapacity {
    /// Tensor storage keeps six products per call and one batch beyond the
    /// submitted window until its last reader retires. Scalar backing comes
    /// from the numerical store's dtype schema, in bytes per product slot.
    #[allow(clippy::too_many_arguments)]
    pub fn for_tokens(
        depth: usize,
        calls: usize,
        requests: usize,
        devices: usize,
        scalar_bytes: u64,
        latent_pool_bytes: u64,
        transfer_bytes_per_read: u64,
    ) -> Result<Self> {
        let window = call_window(depth, calls)?;
        let slots = depth * calls;
        let tensor_store = 6 * (slots + calls);
        let transfer_tickets = slots.min(256);

        Ok(Self {
            latent_pool_bytes,
            tensor_store,
            device_product_bytes: product_bytes(tensor_store, requests, window, scalar_bytes)
                * devices as u64,
            transfer_bytes: transfer_bytes_per_read.max(1) * transfer_tickets as u64,
            transfer_tickets,
            host_lane_inflight: 256,
        })
    }

    /// A media rank retains every local product in full. Read tickets also
    /// cover assembly across all producing ranks, including stateless ranks.
    #[allow(clippy::too_many_arguments)]
    pub fn for_requests(
        depth: usize,
        calls: usize,
        requests: usize,
        ranks: usize,
        scalar_bytes: u64,
        product_bytes_per_request: u64,
        concurrent_imports: usize,
        latent_pool_bytes: u64,
    ) -> Result<Self> {
        let window = request_tensor_window(depth, requests)?;
        let slots = (depth * calls).max(requests * concurrent_imports);
        let tensor_store = 6 * (slots + calls);

        Ok(Self {
            latent_pool_bytes,
            tensor_store,
            device_product_bytes: product_bytes(tensor_store, requests, window, scalar_bytes),
            transfer_bytes: (requests as u64 * product_bytes_per_request).max(1),
            transfer_tickets: ranks.max(slots.min(256)),
            host_lane_inflight: requests * (window + 1),
        })
    }
}

fn product_bytes(products: usize, requests: usize, window: usize, scalar_bytes: u64) -> u64 {
    // One byte per generic value, plus the store's scalar banks. Each request
    // relay row holds two int64 values and two flags, with a retirement lane
    // beyond the unresolved window and slot zero reserved for padding.
    products as u64 * (scalar_bytes + 1) + ((requests + 1) * (window + 1) * 18) as u64
}

/// Device storage for two latent banks, optional step workspace, page indices
/// and a float32 timestep for each one-based request slot.
pub fn latent_pool_bytes(
    requests: usize,
    pages: usize,
    page_units: usize,
    width: usize,
    element_bytes: usize,
    with_workspace: bool,
) -> Result<u64> {
    if requests == 0 || pages < 2 || page_units == 0 || width == 0 || element_bytes == 0 {
        return Err(Error::Invalid("latent pool dimensions are invalid".into()));
    }

    let page_bytes = page_units as u64 * width as u64 * element_bytes as u64;
    let workspace = if with_workspace {
        (pages - 1) as u64 * page_bytes
    } else {
        0
    };
    Ok(2 * pages as u64 * page_bytes
        + workspace
        + (pages - 1) as u64 * 8
        + (requests + 1) as u64 * 4)
}
