use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// A canonical lowercase SHA-256 digest.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Digest(String);

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("digest must be exactly 64 lowercase hexadecimal characters")]
pub struct DigestError;

impl Digest {
    pub const HEX_LEN: usize = 64;

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn zero() -> Self {
        Self("0".repeat(Self::HEX_LEN))
    }

    pub fn into_string(self) -> String {
        self.0
    }

    pub fn from_sha256_bytes(bytes: [u8; 32]) -> Self {
        let mut value = String::with_capacity(Self::HEX_LEN);
        for byte in bytes {
            use fmt::Write as _;
            write!(&mut value, "{byte:02x}").expect("writing to a String is infallible");
        }
        Self(value)
    }

    pub fn validate(value: &str) -> Result<(), DigestError> {
        if value.len() == Self::HEX_LEN
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            Ok(())
        } else {
            Err(DigestError)
        }
    }
}

impl TryFrom<&str> for Digest {
    type Error = DigestError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::validate(value)?;
        Ok(Self(value.to_owned()))
    }
}

impl TryFrom<String> for Digest {
    type Error = DigestError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::validate(&value)?;
        Ok(Self(value))
    }
}

impl AsRef<str> for Digest {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl std::ops::Deref for Digest {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        self.as_str()
    }
}

impl fmt::Display for Digest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl Serialize for Digest {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for Digest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::try_from(value).map_err(serde::de::Error::custom)
    }
}
