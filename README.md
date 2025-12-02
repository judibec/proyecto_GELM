# Pipeline LangChain para reglas de negocio

Esta utilidad permite indexar documentos (PDF, DOCX, TXT, MD) con reglas de negocio en un índice FAISS y generar fragmentos de código a partir de un requerimiento. Usa LangChain con embeddings de OpenAI o Hugging Face.

## Requisitos previos

1. Python 3.10+
2. Instalar dependencias:

   ```bash
   pip install -r requirements.txt
   ```

3. Variables de entorno según el proveedor de embeddings/LLM:
   - **OpenAI**: `OPENAI_API_KEY`
   - **HuggingFace**: `HUGGINGFACEHUB_API_TOKEN` (o token compatible para el modelo elegido)

## Preparar los documentos

1. Copia tus reglas en la carpeta `data/` (o usa `--data_dir` para otra ruta). Se aceptan `.pdf`, `.docx`, `.txt` y `.md`.
2. Ejemplo de estructura:

   ```bash
   data/
   ├─ reglas.pdf
   └─ anexos.txt
   ```

## Crear el índice vectorial

Ejecuta la ingesta para procesar los documentos y guardar el índice FAISS:

```bash
python main.py --ingest --data_dir data --vector_dir vectores/prueba \
  --embedding_provider openai --chunk_size 1200 --chunk_overlap 100
```

- Cambia `--embedding_provider hf` para usar Hugging Face y añade `--embedding_model` si quieres un modelo específico.
- El índice se guarda en `vectores/prueba` por defecto.

## Generar código a partir de un requerimiento

Tienes dos opciones para ingresar el requerimiento:

1. **Texto directo**:
   ```bash
   python main.py --requirement "Crear un endpoint REST para alta de clientes" \
     --vector_dir vectores/prueba
   ```

2. **Archivo de texto/Markdown** (útil si copiaste un requerimiento largo desde Word o PDF a un `.txt`):
   ```bash
   python main.py --requirement_file requerimientos/alta_cliente.txt \
     --vector_dir vectores/prueba
   ```

Notas:
- Si usas `--requirement_file`, el contenido del archivo se envía completo al modelo.
- Puedes combinar `--ingest` y `--requirement` en una misma ejecución para reindexar y generar código en un solo paso.

## Limpieza y ubicación de resultados

- El índice FAISS queda en la carpeta indicada por `--vector_dir`.
- El fragmento generado se imprime en consola. Si quieres guardarlo, redirígelo a un archivo:

  ```bash
  python main.py --requirement "Crear CRUD de productos" --vector_dir vectores/prueba > snippet.py
  ```

## Solución de problemas

- Si ves `OPENAI_API_KEY no está definido`, exporta la variable de entorno antes de ejecutar.
- Verifica que `data/` contenga archivos soportados; de lo contrario la ingesta fallará.
- Para documentos muy grandes, ajusta `--chunk_size` y `--chunk_overlap` para mejorar el contexto recuperado.