//! Shared device budgets and allocation pools for captured numerical calls.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use indexmap::IndexMap;

use crate::{Error, Result};

/// The numerical allocator's pool handle and its backing owner.
pub struct GraphPool<P> {
    pub device: i32,
    pub id: (u64, u64),
    pub value: P,
}

/// A numerical execution and the pools retained for its captures.
pub struct GraphPools<O, P> {
    pub owner: O,
    pub pools: Vec<Arc<GraphPool<P>>>,
}

/// Pool sharing, startup residency and byte limits across numerical owners.
///
/// Capture catalogs remain with the numerical backend. It supplies allocator
/// snapshots and process usage at preparation boundaries; replay needs no
/// allocator inspection. Owners sharing pools must serialize graph replay.
pub struct GraphStorage<O, P> {
    owners: IndexMap<usize, GraphPools<O, P>>,
    budgets: BTreeMap<i32, u64>,
    outside: BTreeMap<i32, i128>,
}

impl<O, P> GraphStorage<O, P> {
    pub fn new(budgets: BTreeMap<i32, u64>) -> Self {
        Self {
            owners: IndexMap::new(),
            budgets,
            outside: BTreeMap::new(),
        }
    }

    /// Borrow shared pools before allocating the remaining devices.
    /// The caller commits only after all numerical allocations succeed.
    pub fn prepare(&self, owner: usize, share: Option<usize>) -> Result<Vec<Arc<GraphPool<P>>>> {
        if self.owners.contains_key(&owner) {
            return Err(Error::State("graph storage owner is already registered"));
        }

        match share {
            Some(share) => self
                .owners
                .get(&share)
                .map(|owner| owner.pools.clone())
                .ok_or(Error::State("shared graph storage owner is not registered")),
            None => Ok(Vec::new()),
        }
    }

    /// Install successfully allocated pools and defaults for unbudgeted devices.
    pub fn commit(
        &mut self,
        key: usize,
        owner: O,
        pools: Vec<Arc<GraphPool<P>>>,
        defaults: impl IntoIterator<Item = (i32, u64)>,
    ) {
        for (device, amount) in defaults {
            self.budgets.entry(device).or_insert(amount);
        }
        self.owners.insert(key, GraphPools { owner, pools });
    }

    pub fn pools(&self, owner: usize) -> Result<&[Arc<GraphPool<P>>]> {
        self.owners
            .get(&owner)
            .map(|owner| owner.pools.as_slice())
            .ok_or(Error::State("graph storage owner is not registered"))
    }

    pub fn owners(&self) -> impl Iterator<Item = &GraphPools<O, P>> {
        self.owners.values()
    }

    pub fn owner(&self, key: usize) -> &O {
        &self.owners[&key].owner
    }

    pub fn has_pools(&self) -> bool {
        self.owners.values().any(|owner| !owner.pools.is_empty())
    }

    pub fn bound_devices(&self) -> impl Iterator<Item = i32> + '_ {
        self.outside.keys().copied()
    }

    pub fn needs_pool_sizes(&self) -> bool {
        self.budgets.len() > self.outside.len()
    }

    /// Attribute a shared pool once, to its first remaining numerical owner.
    pub fn owner_bytes(&self, segments: &[(i32, (u64, u64), u64)]) -> BTreeMap<(usize, i32), u64> {
        let mut owners = HashMap::new();
        for (&key, owner) in &self.owners {
            for pool in &owner.pools {
                owners.entry((pool.device, pool.id)).or_insert(key);
            }
        }

        let mut sizes = BTreeMap::new();
        for &(device, id, bytes) in segments {
            if let Some(&owner) = owners.get(&(device, id)) {
                *sizes.entry((owner, device)).or_default() += bytes;
            }
        }
        sizes
    }

    pub fn pool_bytes(&self, segments: &[(i32, (u64, u64), u64)]) -> BTreeMap<i32, u64> {
        let mut sizes: BTreeMap<_, _> = self.budgets.keys().map(|&device| (device, 0)).collect();
        for ((_, device), used) in self.owner_bytes(segments) {
            *sizes.entry(device).or_default() += used;
        }
        sizes
    }

    /// Bind the process footprint outside graph pools at the start of capture.
    pub fn set_budget(&mut self, device: i32, amount: u64, process: u64, pooled: u64) {
        self.outside
            .insert(device, i128::from(process) - i128::from(pooled));
        self.budgets.insert(device, amount);
    }

    /// Account for process growth while preparing, and only pools after seal.
    pub fn resident_bytes(
        &self,
        pools: &BTreeMap<i32, u64>,
        mut process: impl FnMut(i32) -> u64,
    ) -> BTreeMap<i32, i128> {
        self.budgets
            .keys()
            .map(|&device| {
                let used = match self.outside.get(&device) {
                    Some(&outside) => i128::from(process(device)) - outside,
                    None => i128::from(pools.get(&device).copied().unwrap_or(0)),
                };
                (device, used)
            })
            .collect()
    }

    pub fn check(&self, residency: &BTreeMap<i32, i128>) -> Result<()> {
        for (&device, &used) in residency {
            let budget = self.budgets[&device];
            if used > i128::from(budget) {
                return Err(Error::Invalid(format!(
                    "graph residency on cuda:{device} exceeds its byte budget ({used}>{budget})"
                )));
            }
        }
        Ok(())
    }

    /// Serving growth belongs to the worker grant, not the sealed graph pools.
    pub fn seal(&mut self) {
        self.outside.clear();
    }

    /// The caller drains graphs and views before releasing their pool owner.
    pub fn release(&mut self, owner: usize) -> Option<GraphPools<O, P>> {
        self.owners.shift_remove(&owner)
    }

    pub fn close(&mut self) -> Vec<GraphPools<O, P>> {
        self.outside.clear();
        self.owners.drain(..).map(|(_, owner)| owner).collect()
    }
}

/// The worker reserves the same graph share when planning and allocating.
pub fn graph_storage_budget_bytes(total_device_bytes: i64) -> u64 {
    (total_device_bytes.max(0) as f64 * 0.10) as u64
}
