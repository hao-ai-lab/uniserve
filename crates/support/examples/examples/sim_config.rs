fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = uniserve_examples::sim_http_config("sim-model");
    println!("{}", serde_json::to_string_pretty(&config)?);
    Ok(())
}
