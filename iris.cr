require "shainet"
require "csv"

  # Configure label encoding
  label = {
    "setosa"     => [0.to_f64, 0.to_f64, 1.to_f64],
    "versicolor" => [0.to_f64, 1.to_f64, 0.to_f64],
    "virginica"  => [1.to_f64, 0.to_f64, 0.to_f64],
  }

  # Initiate a new network
  iris = SHAInet::Network.new
  iris.add_layer(:input, 4, :memory, SHAInet.sigmoid)
  iris.add_layer(:hidden, 5, :memory, SHAInet.sigmoid)
  iris.add_layer(:output, 3, :memory, SHAInet.sigmoid)
  iris.fully_connect

  # load all relevant information from the iris.csv
  inputs = Array(Array(Float64)).new
  outputs = Array(Array(Float64)).new
  CSV.each_row(File.read(File.join(__DIR__, "data", "iris.csv"))) do |row|
    row_arr = Array(Float64).new
    row[0..-2].each do |num|
      row_arr << num.to_f64
    end
    inputs << row_arr
    outputs << label[row[-1]]
  end

  iris.graph(title: "Iris Dataset",
             labels: { :input => "Input",
                       :hidden => "activation",
                       :output => "error" })

  normalized = SHAInet::TrainingData.new(inputs, outputs)
  normalized.normalize_min_max
  iris.train_batch(data: normalized.data,
                   training_type: :adam,
                   cost_function: :mse,
                   epochs: 20000,
                   error_threshold: 0.0001)

