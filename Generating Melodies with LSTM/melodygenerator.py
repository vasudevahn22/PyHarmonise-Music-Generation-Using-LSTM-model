# import json
# import numpy as np
# import tensorflow.keras as keras
# from preprocess import SEQUENCE_LENGTH, MAPPING_PATH

# class MelodyGenerator:
#     """A class that wraps the LSTM model and offers utilities to generate melodies."""

#     def __init__(self, model_path="model.h5"):
#         """Constructor that initialises TensorFlow model"""

#         self.model_path = model_path
#         self.model = keras.models.load_model(model_path)

#         with open(MAPPING_PATH, "r") as fp:
#             self._mappings = json.load(fp)

#         self._start_symbols = ["/"] * SEQUENCE_LENGTH


#     def generate_melody(self, seed, num_steps, max_sequence_length, temperature):
#         """Generates a melody using the DL model and returns a midi file.

#         :param seed (str): Melody seed with the notation used to encode the dataset
#         :param num_steps (int): Number of steps to be generated
#         :param max_sequence_len (int): Max number of steps in seed to be considered for generation
#         :param temperature (float): Float in interval [0, 1]. Numbers closer to 0 make the model more deterministic.
#             A number closer to 1 makes the generation more unpredictable.

#         :return melody (list of str): List with symbols representing a melody
#         """

#         # create seed with start symbols
#         seed = seed.split()
#         melody = seed
#         seed = self._start_symbols + seed

#         # map seed to int
#         seed = [self._mappings[symbol] for symbol in seed]

#         for _ in range(num_steps):

#             # limit the seed to max_sequence_length
#             seed = seed[-max_sequence_length:]

#             # one-hot encode the seed
#             onehot_seed = keras.utils.to_categorical(seed, num_classes=len(self._mappings))
#             # (1, max_sequence_length, num of symbols in the vocabulary)
#             onehot_seed = onehot_seed[np.newaxis, ...]

#             # make a prediction
#             probabilities = self.model.predict(onehot_seed)[0]
#             # [0.1, 0.2, 0.1, 0.6] -> 1
#             output_int = self._sample_with_temperature(probabilities, temperature)

#             # update seed
#             seed.append(output_int)

#             # map int to our encoding
#             output_symbol = [k for k, v in self._mappings.items() if v == output_int][0]

#             # check whether we're at the end of a melody
#             if output_symbol == "/":
#                 break

#             # update melody
#             melody.append(output_symbol)

#         return melody


#     def _sample_with_temperature(self, probabilites, temperature):
#         """Samples an index from a probability array reapplying softmax using temperature

#         :param predictions (nd.array): Array containing probabilities for each of the possible outputs.
#         :param temperature (float): Float in interval [0, 1]. Numbers closer to 0 make the model more deterministic.
#             A number closer to 1 makes the generation more unpredictable.

#         :return index (int): Selected output symbol
#         """
#         predictions = np.log(probabilites) / temperature
#         probabilites = np.exp(predictions) / np.sum(np.exp(predictions))

#         choices = range(len(probabilites)) # [0, 1, 2, 3]
#         index = np.random.choice(choices, p=probabilites)

#         return index


# if __name__ == "__main__":
#     mg = MelodyGenerator()
#     seed = "55 _ _ _ 60 _ _ _ 55 _ _ _ 55 _"
#     melody = mg.generate_melody(seed, 500, SEQUENCE_LENGTH, 0.7)
#     print(melody)

import json
import numpy as np
import tensorflow.keras as keras
from preprocess import SEQUENCE_LENGTH, MAPPING_PATH


class MelodyGenerator:
    """A class that wraps the LSTM model and offers utilities to generate melodies with advanced mathematical adjustments."""

    def __init__(self, model_path="model.h5"):
        """Constructor that initializes the TensorFlow model and mappings."""

        self.model_path = model_path
        self.model = keras.models.load_model(model_path)

        # Load symbol-to-integer mappings
        with open(MAPPING_PATH, "r") as fp:
            self._mappings = json.load(fp)

        self._start_symbols = ["/"] * SEQUENCE_LENGTH

        # Initialize weights for symbol bias adjustment (example: rests might be less likely)
        self.symbol_weights = self._initialize_symbol_weights()

    def _initialize_symbol_weights(self):
        """Assign weights or biases to each symbol."""
        weights = {}
        for symbol in self._mappings.keys():
            # Example weights: penalize rests slightly more
            weights[symbol] = 0.9 if symbol == "r" else 1.0
        return weights

    def _calculate_entropy(self, probabilities):
        """Calculate Shannon entropy of the probability distribution."""
        return -np.sum(probabilities * np.log(probabilities + 1e-10))  # Add small epsilon to avoid log(0)

    def _adjust_probabilities(self, probabilities, temperature, entropy):
        """Adjust probabilities using temperature and bias."""
        # Apply temperature scaling
        scaled_logits = np.log(probabilities) / temperature
        scaled_probabilities = np.exp(scaled_logits) / np.sum(np.exp(scaled_logits))

        # Apply symbol bias adjustment
        adjusted_probabilities = np.array([
            scaled_probabilities[i] * self.symbol_weights[symbol]
            for i, symbol in enumerate(self._mappings.keys())
        ])

        # Renormalize after bias adjustment
        adjusted_probabilities /= np.sum(adjusted_probabilities)

        return adjusted_probabilities

    def generate_melody(self, seed, num_steps, max_sequence_length, initial_temperature):
        """
        Generates a melody using the DL model with dynamic adjustments.

        :param seed (str): Initial sequence of symbols to start the melody.
        :param num_steps (int): Total number of steps to generate.
        :param max_sequence_length (int): Maximum number of steps considered in the sequence window.
        :param initial_temperature (float): Starting temperature for randomness control.
        :return melody (list of str): Generated melody as a sequence of symbols.
        """

        seed_sequence = seed.split()
        melody = seed_sequence.copy()
        padded_seed = self._start_symbols + seed_sequence
        seed_encoded = [self._mappings[symbol] for symbol in padded_seed]

        temperature = initial_temperature

        for _ in range(num_steps):
            sequence_window = seed_encoded[-max_sequence_length:]
            onehot_encoded_sequence = keras.utils.to_categorical(
                sequence_window, num_classes=len(self._mappings)
            )[np.newaxis, ...]

            # Predict probabilities for the next symbol
            probabilities = self.model.predict(onehot_encoded_sequence)[0]

            # Calculate entropy of probabilities
            entropy = self._calculate_entropy(probabilities)

            # Dynamically adjust temperature based on entropy
            if entropy < 1.0:  # Low entropy: increase randomness
                temperature = min(temperature + 0.1, 1.0)
            elif entropy > 2.5:  # High entropy: decrease randomness
                temperature = max(temperature - 0.1, 0.5)

            # Adjust probabilities using temperature and bias
            adjusted_probabilities = self._adjust_probabilities(probabilities, temperature, entropy)

            # Sample an output index
            output_index = np.random.choice(range(len(adjusted_probabilities)), p=adjusted_probabilities)

            seed_encoded.append(output_index)
            output_symbol = [symbol for symbol, index in self._mappings.items() if index == output_index][0]

            if output_symbol == "/":
                break

            melody.append(output_symbol)

        return melody


if __name__ == "__main__":
    melody_generator = MelodyGenerator()
    seed_sequence = "55 _ _ _ 60 _ _ _ 55 _ _ _ 55 _"
    generated_melody = melody_generator.generate_melody(
        seed=seed_sequence, num_steps=500, max_sequence_length=SEQUENCE_LENGTH, initial_temperature=0.7
    )
    print(generated_melody)


















