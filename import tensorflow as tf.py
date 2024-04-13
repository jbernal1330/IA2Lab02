import tensorflow as tf

# Verifica si TensorFlow tiene acceso a la GPU
print("GPUs disponibles: ", tf.config.experimental.list_physical_devices('GPU'))

# Configura TensorFlow para que use la GPU si está disponible
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Establece la memoria de la GPU para que se asigne a demanda y no toda de una vez (opcional)
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Un error de tiempo de ejecución aquí significa que ya se han creado los objetos del programa
    print(e)