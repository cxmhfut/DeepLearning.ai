from course5.week_01.Test.dinosaur_island_character_level_language_model.shakespeare_utils import *

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

model.fit(x, y, batch_size=128, epochs=1, callbacks=[print_callback])

# Run this cell to try with different inputs without having to re-train the models
generate_output()