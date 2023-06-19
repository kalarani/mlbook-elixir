Mix.install([
  {:csv, "~> 3.0"}
])

defmodule LinearRegression do
  import CSV

  def compute_model_output(x, w, b) do
    Enum.map(x, fn i -> compute_output_for(i, w, b) end)
  end

  def compute_output_for(i, w, b) do
    w * i + b
  end

  def compute_cost(x, y, w, b) do
    m = Enum.count(x)

    cost =
      x
      |> Enum.with_index()
      |> Enum.map(fn {i, index} -> (compute_output_for(i, w, b) - Enum.at(y, index)) ** 2 end)
      |> Enum.sum()

    total_cost = 1 / (2 * m) * cost
    total_cost
  end

  def gradient_descent_step(x, y, w, b) do
    m = Enum.count(x)
    dj_dw = x
    |> Enum.with_index()
    |> Enum.map(fn {i, index} -> (compute_output_for(i, w, b) - Enum.at(y, index)) * i end)
    |> Enum.sum()

    dj_db = x
    |> Enum.with_index()
    |> Enum.map(fn {i, index} -> (compute_output_for(i, w, b) - Enum.at(y, index)) end)
    |> Enum.sum()

    [w - dj_dw, b - dj_db]
  end

  def read_dataset(filename) do
    filename
    |> Path.expand(__DIR__)
    |> File.stream!
    |> CSV.decode()
    |> Enum.drop(1) # skip header
    |> Enum.map(fn t -> elem(t, 1) end)
  end

  def train_from(filename) do
    raw_dataset = read_dataset(filename)
    x = raw_dataset |> Enum.map(fn d -> elem(Integer.parse(Enum.at(d, 0)),0) end)
    y = raw_dataset |> Enum.map(fn d -> elem(Float.parse(Enum.at(d, 1)),0) end)
    training_dataset = %{x: x, y: y}
    w = 10 # initial value for w
    b = 10 # initial value for b
    n = 50 # number of iterations

    [w, b] = gradient_descent_loop(x, y, w, b, n)

    IO.inspect("End values for w: #{w}, b: #{b}");
  end

  def gradient_descent_loop(x, y, w, b, n) when n > 1 do
    [w1, b1] = gradient_descent_step(x, y, w, b)
    IO.inspect("Iteration #{n}: #{w1}, #{b1}")
    gradient_descent_loop(x, y, w1, b1, n-1)
  end

  def gradient_descent_loop(x, y, w, b, 1) do
    gradient_descent_step(x, y, w, b)
  end
end

LinearRegression.train_from("linear_regression_dataset.csv")
