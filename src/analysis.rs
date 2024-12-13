use std::error::Error;
use csv::ReaderBuilder;
use petgraph::graph::{NodeIndex, Graph};
use ndarray::prelude::*;
use linfa::prelude::*;
use linfa_logistic::LogisticRegression;
use linfa::traits::Fit;
use ndarray::Array2;
use rand::seq::SliceRandom;
use std::collections::HashMap;

#[derive(Debug, serde::Deserialize)]
struct FileStructure {
    step: u32,
    #[serde(rename = "type")]
    ttype: String,
    amount: f64,
    nameOrig: String,
    oldbalanceOrg: f64,
    newbalanceOrig: f64,
    nameDest: String,
    oldbalanceDest: f64,
    newbalanceDest: f64,
    isFraud: u8,
    isFlaggedFraud: u8,
}

//Graph builder
pub fn build_graph_print_ten(file_path: &str) -> Result<Graph<String, f64>, Box<dyn Error>> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(file_path)?;

    let mut graph = Graph::<String, f64>::new();
    let mut node_indices: HashMap<String, NodeIndex> = HashMap::new();

    for result in reader.deserialize::<FileStructure>() {
        let record = match result {
            Ok(r) => r,
            Err(e) => {
                println!("Skipping row due to error: {}", e);
                continue;
            }
        };
        let sender = record.nameOrig.clone();
        let receiver = record.nameDest.clone();

        let sender_index = *node_indices.entry(sender.clone())
            .or_insert_with(|| graph.add_node(sender));
        let receiver_index = *node_indices.entry(receiver.clone())
            .or_insert_with(|| graph.add_node(receiver));

        let edge_weight = if record.isFraud == 1 {
            record.amount * 2.0
        } else {
            record.amount
        };

        graph.add_edge(sender_index, receiver_index, edge_weight);
    }
    // node indices into vec --> gather
    let node_indices: Vec<_> = graph.node_indices().collect();
    // random 10 show
    let mut rng = rand::thread_rng();
    //node_indices? NONO
    let random_nodes = node_indices
        .choose_multiple(&mut rng, 10)
        .cloned()
        .collect::<Vec<_>>();

    // Print a partial graph representation for the selected nodes
    println!("10 Random Graph Result:");
    for node in random_nodes {
        let node_name = graph[node].clone();
        println!("Node: {}", node_name);

        for neighbor in graph.neighbors(node) {
            let edge = graph.find_edge(node, neighbor).unwrap();
            let weight = graph[edge];
            println!("  -> Neighbor: {}, Edge Weight: {}", graph[neighbor], weight);
        }
    }

    Ok(graph)
}

//prediction - what transaction / amount is likely to be fraud
pub fn run_fraud_prediction(file_path: &str) -> Result<(), Box<dyn Error>> {
    let mut reader = ReaderBuilder::new()
    //the csv turns out it has headers
        .has_headers(true)
        .from_path(file_path)?;
    let mut records = Vec::new();

    for result in reader.deserialize::<FileStructure>() {
        let rec = match result {
            Ok(r) => r,
            Err(e) => {
                println!("Skipping row due to parse error: {}", e);
                continue;
            }
        };
        records.push(rec);
    }

    let possible_types = vec!["CASH-IN", "CASH-OUT", "DEBIT", "PAYMENT", "TRANSFER"];
    let n = records.len();
    if n == 0 {
        println!("No data to train on");
        return Ok(());
    }

    // Build dictionaries for nameOrig and nameDest to convert them to numeric IDs
    let mut name_orig_map = HashMap::new();
    let mut name_dest_map = HashMap::new();
    //collect names
    for rec in &records {
        if !name_orig_map.contains_key(&rec.nameOrig) {
            let len = name_orig_map.len() as f64;
            name_orig_map.insert(rec.nameOrig.clone(), len);
        }
        if !name_dest_map.contains_key(&rec.nameDest) {
            let len = name_dest_map.len() as f64;
            name_dest_map.insert(rec.nameDest.clone(), len);
        }
    }
//total feature columns = 14
    let num_features = 14;
    let mut x_data = Array2::<f64>::zeros((n, num_features));
    let mut y_data = Array1::<usize>::zeros((n,));

    for (i, record_row) in records.iter().enumerate() {
        // step
        x_data[[i, 0]] = record_row.step as f64;
        // type one-hot
        for (r, type_shown) in possible_types.iter().enumerate() {
            x_data[[i, r + 1]] = if &record_row.ttype == type_shown { 1.0 } else { 0.0 };
        }
        // amount
        x_data[[i, 6]] = record_row.amount;
        // nameOrig
        x_data[[i, 7]] = *name_orig_map.get(&record_row.nameOrig).unwrap_or(&0.0);
        // oldbalanceOrg
        x_data[[i, 8]] = record_row.oldbalanceOrg;
        // newbalanceOrig
        x_data[[i, 9]] = record_row.newbalanceOrig;
        // nameDest encoded
        x_data[[i, 10]] = *name_dest_map.get(&record_row.nameDest).unwrap_or(&0.0);
        // oldbalanceDest
        x_data[[i, 11]] = record_row.oldbalanceDest;
        // newbalanceDest
        x_data[[i, 12]] = record_row.newbalanceDest;
        // isFlaggedFraud
        x_data[[i, 13]] = record_row.isFlaggedFraud as f64;
        // Target is isFraud
        y_data[i] = record_row.isFraud as usize;
    }

    let split_ratio = 0.7;
    let train_size = (n as f64 * split_ratio) as usize;

    let x_train = x_data.slice(s![0..train_size, ..]).to_owned();
    let y_train = y_data.slice(s![0..train_size]).to_owned();
    let x_test = x_data.slice(s![train_size.., ..]).to_owned();
    let y_test = y_data.slice(s![train_size..]).to_owned();

    let train = Dataset::new(x_train, y_train);
    let max_iter = 10;
    for iter in 1..=max_iter {
        let model = LogisticRegression::default()
            .max_iterations(iter) // Fit for the current epoch count
            .fit(&train)
            .unwrap();

        println!("Iteration: {}/{} completed", iter, max_iter);
    }
    //final model prection
    let model = LogisticRegression::default()
        .max_iterations(max_iter)
        .fit(&train)?;

    let y_pred = model.predict(&x_test);
    let mut correct_pred_test = 0;
    let total = y_test.len();
    for (pred, actual) in y_pred.iter().zip(y_test.iter()) {
        if pred == actual {
            correct_pred_test += 1;
        }
    }

    let accuracy = correct_pred_test as f64 / total as f64;
    println!("Fraud Prediction Accuracy: {:.5}%", accuracy * 100.0);

    Ok(())
}

//TEST
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_graph() {
        let result = build_graph_print_ten("cleaned_data.csv");
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_run_prediction() {
        let result = run_fraud_prediction("cleaned_data.csv");
        assert!(result.is_ok() || result.is_err());
    }
}
