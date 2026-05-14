# Cambios principales para la reentrega

- Tipos en las cabeceras de las funciones y mejora de docstrings
- Uso de Pydantic
- Mejora de los prompts inicial y de procesamiento de carta
- Si se envía paquete, se envía carta de confirmación de envío excepto si el paquete se envía en respuesta a otra carta de confirmación, para no entrar en un bucle de cartas de confirmación (comprobamos que el asunto no incluya "confirm", "enviad"... para mandar carta de confirmación).
- Registro de logs con logurus
- Generación de contexto dinámico en los prompts que se adapta a la situación actual:
  - Si está cerca del objetivo, le avisa para que conseguir los pocos recursos faltantes sea la prioridad máxima
  - Si faltan/sobran muchos recursos
  - Si se ha quedado sin recursos sobrantes le dice que rechace todos los tratos
  - También, si el modelo de Ollama es pequeño, le avisa de la tendencia a alucinar
- Se confunde con enviar/recibir: mejora del prompt procesar_carta
- Estructuras de datos específicas para cartas y estado de recursos