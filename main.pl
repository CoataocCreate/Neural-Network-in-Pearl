use strict;
use warnings;
use List::Util qw(sum);

# Sigmoid activation function
sub sigmoid {
    my ($x) = @_;
    return 1.0 / (1.0 + exp(-$x));
}

# Derivative of sigmoid function
sub sigmoid_derivative {
    my ($x) = @_;
    return $x * (1.0 - $x);
}

# Initialize weights with random values
sub initialize_weights {
    my ($size) = @_;
    my @weights;
    for (1..$size) {
        push @weights, rand() * 0.2 - 0.1; # Small random values
    }
    return \@weights;
}

# Define neural network
sub train {
    my ($inputs, $expected_outputs, $epochs, $learning_rate) = @_;
    
    my $input_size = @{$inputs->[0]};
    my $hidden_size = 4;
    my $output_size = @{$expected_outputs->[0]};
    
    my $input_hidden_weights = initialize_weights($input_size * $hidden_size);
    my $hidden_output_weights = initialize_weights($hidden_size * $output_size);
    
    for my $epoch (1..$epochs) {
        for my $i (0..$#$inputs) {
            my $input = $inputs->[$i];
            my $expected_output = $expected_outputs->[$i];
            
            # Forward propagation
            my @hidden_layer;
            my @output_layer;
            my @hidden_layer_input;
            
            # Calculate hidden layer
            for my $j (0..$hidden_size-1) {
                $hidden_layer_input[$j] = sum(map { $input->[$_] * $input_hidden_weights->[$j * $input_size + $_] } 0..$input_size-1);
                $hidden_layer[$j] = sigmoid($hidden_layer_input[$j]);
            }
            
            # Calculate output layer
            for my $j (0..$output_size-1) {
                $output_layer[$j] = sum(map { $hidden_layer[$_] * $hidden_output_weights->[$j * $hidden_size + $_] } 0..$hidden_size-1);
                $output_layer[$j] = sigmoid($output_layer[$j]);
            }
            
            # Compute errors
            my @output_error = map { $expected_output->[$_] - $output_layer[$_] } 0..$output_size-1;
            my @hidden_error;
            
            for my $j (0..$hidden_size-1) {
                $hidden_error[$j] = sum(map { $output_error[$_] * $hidden_output_weights->[$_ * $hidden_size + $j] } 0..$output_size-1);
                $hidden_error[$j] *= sigmoid_derivative($hidden_layer[$j]);
            }
            
            # Update weights
            for my $j (0..$output_size-1) {
                for my $k (0..$hidden_size-1) {
                    $hidden_output_weights->[$j * $hidden_size + $k] += $learning_rate * $output_error[$j] * $hidden_layer[$k];
                }
            }
            
            for my $j (0..$hidden_size-1) {
                for my $k (0..$input_size-1) {
                    $input_hidden_weights->[$j * $input_size + $k] += $learning_rate * $hidden_error[$j] * $input->[$k];
                }
            }
        }
    }
    
    return ($input_hidden_weights, $hidden_output_weights);
}

sub predict {
    my ($input, $input_hidden_weights, $hidden_output_weights) = @_;
    
    my $input_size = @{$input};
    my $hidden_size = 4;
    my $output_size = 1;
    
    my @hidden_layer;
    my @output_layer;
    
    # Calculate hidden layer
    my @hidden_layer_input;
    for my $j (0..$hidden_size-1) {
        $hidden_layer_input[$j] = sum(map { $input->[$_] * $input_hidden_weights->[$j * $input_size + $_] } 0..$input_size-1);
        $hidden_layer[$j] = sigmoid($hidden_layer_input[$j]);
    }
    
    # Calculate output layer
    for my $j (0..$output_size-1) {
        $output_layer[$j] = sum(map { $hidden_layer[$_] * $hidden_output_weights->[$j * $hidden_size + $_] } 0..$hidden_size-1);
        $output_layer[$j] = sigmoid($output_layer[$j]);
    }
    
    return \@output_layer;
}

# Define XOR problem data
my @inputs = (
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
);

my @expected_outputs = (
    [0],
    [1],
    [1],
    [0]
);

# Train the network
my ($input_hidden_weights, $hidden_output_weights) = train(\@inputs, \@expected_outputs, 10000, 0.1);

# Test the network
for my $i (0..$#inputs) {
    my $output = predict($inputs[$i], $input_hidden_weights, $hidden_output_weights);
    print "Input: @{$inputs[$i]} => Output: @$output\n";
}
