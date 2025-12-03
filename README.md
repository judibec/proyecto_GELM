# Generación Automática de Código Basado en Reglas de Negocio con RAG
## Problemática
La Recuperación Aumentada por Generación **(RAG)** es una técnica que combina modelos de lenguaje con búsqueda de información externa, permitiendo que un modelo generativo consulte una base de conocimientos antes de producir código o respuestas
En este proyecto se busca automatizar la generación de código utilizando RAG para interpretar reglas de negocio definidas en documentos (PDF, Word, texto plano) y producir fragmentos de código fuente alineados con dichas reglas. El caso de estudio se centra en la industria de préstamos de crédito, donde las reglas de aprobación o rechazo de un préstamo están formalizadas en documentos. Por ejemplo, se manejan condiciones como:
* Estado del préstamo debe estar en [PENDIENTE, APROBADO, EN PROCESO, TERMINADO].
* Edad del solicitante debe ser mayor a 18 años.
* Puntaje de crédito del solicitante superior a 4.5.
* Monto de préstamo solicitado menor a 5 millones.
Dichas reglas deben traducirse consistentemente a validaciones en el código de las aplicaciones. Para lograrlo, el proyecto propone un modelo de IA capaz de leer y entender estas reglas de negocio desde los documentos originales, y luego generar automáticamente código fuente que cumpla con ellas en el lenguaje de programación deseado.

En resumen, el objetivo es crear un sistema que garantice que el código generado refleje fielmente las políticas de la empresa. El **objetivo general** es *desarrollar un modelo de IA que interprete reglas de negocio desde documentos y genere código fuente alineado automáticamente a esas reglas*. Para alcanzar este fin, se plantean tres objetivos específicos:
1. Implementar un sistema RAG que ingiera e indexe los documentos con las reglas de negocio, convirtiéndolos en vectores de embeddings para facilitar su recuperación.
2. Integrar un modelo generativo avanzado (p. ej., GPT-4) que, dado un requerimiento del usuario, recupere las reglas pertinentes y produzca un fragmento de código que satisfaga dicho requerimiento respetando las reglas.
3. Evaluar el rendimiento del modelo generando código en múltiples lenguajes de programación (específicamente Python y Java) para comprobar que las reglas se aplican correctamente sin depender de un solo lenguaje.

## Trabajos relacionados

*Ilustración conceptual de RAG aplicado a la generación de código.*

La aplicación de RAG para generación de código ha ganado interés recientemente en la industria del software. Diversos autores destacan que combinar recuperación de conocimiento con modelos generativos permite crear código más preciso y adaptado al contexto específico. Por ejemplo, Arooj (2025) señala que RAG no solo automatiza la escritura de código, sino que lo personaliza al contexto del proyecto, generando resultados que parecen escritos por alguien familiarizado con el sistema.
Además, esta técnica ayuda a mantener la consistencia y calidad del código al reutilizar componentes y seguir estándares internos, reduciendo la deuda técnica como si se tuviera un desarrollador senior supervisando 24/7. 
En cuanto a implementaciones prácticas, Singh (2024) demostró un pipeline con LangChain y LlamaIndex que utiliza **FAISS** como base vectorial y un modelo Code Llama local, logrando un generador de código impulsado por RAG.
Incluso se ha propuesto que los LLM pueden interpretar reglas de negocio descritas en lenguaje natural y traducirlas automáticamente a configuraciones o código ejecutable, facilitando que equipos no técnicos apliquen cambios en la lógica empresarial.
Estos trabajos relacionados respaldan la viabilidad del enfoque adoptado en este proyecto.

### Referencias

1. Arooj. “How To Use RAG for Code Generation.” Chitika Tech Blog, 3 Feb 2025 [chikita.com](https://www.chitika.com/rag-for-code-generation/#:~:text=Retrieval,knows%20your%20system%20inside%20out)
2. Samar Singh. “Code Retrieval and Generation with LangChain & LlamaIndex: RAG & AI Agents (With Demo Projects).” Medium, 22 May 2024. [medium.com](https://medium.com/@samarrana407/code-retrieval-and-generation-with-langchain-llamaindex-rag-ai-agents-with-demo-projects-6db7a5ed79c6)
3. Nitin Rachabathuni. “How LLMs Can Handle Business Logic and Rules Effectively.” Medium, 23 Jul 2025 [https://nitin-rachabathuni.medium.com](https://nitin-rachabathuni.medium.com/how-llms-can-handle-business-logic-and-rules-effectively-b9e1a6b185b6)
4. AWS. “What is Retrieval-Augmented Generation (RAG)?” AWS Machine Learning Blog [aws.amazon.com](https://aws.amazon.com/what-is/retrieval-augmented-generation/#:~:text=What%20is%20Retrieval)

## Metodologia

La solución se implementó mediante un pipeline de LangChain que combina ingestión de documentos, búsqueda semántica y generación de código. Primero, un componente ingestor (DocumentIngestor) lee los documentos de reglas de negocio (en formatos PDF, DOCX o texto), los divide en fragmentos manejables y genera representaciones vectoriales (embeddings) para cada fragmento. Estos embeddings se almacenan en un índice vectorial usando FAISS (Facebook AI Similarity Search), permitiendo recuperar rápidamente fragmentos relevantes dada una consulta. Luego, un componente generador (CodeGenerator) carga el índice vectorial y, ante una petición del usuario, utiliza un modelo generativo (por ejemplo GPT-4 a través de la API de OpenAI) para producir código. Este modelo se integra mediante la clase RetrievalQA de LangChain: al recibir una pregunta o requerimiento, primero recupera los fragmentos de reglas de negocio pertinentes (contexto) y luego alimenta tanto el contexto como la pregunta al modelo de lenguaje. El prompt utilizado instruye al modelo a generar únicamente código (p. ej., una función o segmento) acompañado de comentarios que expliquen cómo cada parte del código satisface las reglas.

Para ejecutar el proyecto remitase a [Getting_Start](/Getting_Start.md)

## Resultados y Discusión

Los experimentos iniciales demostraron que el sistema puede generar fragmentos de código válidos en Python y Java a partir de las mismas reglas de negocio. Por ejemplo, dada la regla de validación de edad (>18 años), el modelo fue capaz de producir correctamente una sentencia condicional if tanto en Python (usando sintaxis Pythonic) como en Java (empleando la sintaxis propia de Java), respetando en ambos casos la lógica de negocio. Estos resultados indican una capacidad multilenguaje del enfoque propuesto, y sugieren que el modelo comprende las reglas de forma abstracta, aplicándolas luego en el idioma de programación solicitado.

Durante los experimentos la reglas de negocio se modificaron y se pudo notar que el moedelo respondia satisfactoriamente al cambio de estas, puesto que al mismo prompt respondia actualizando el codigo, incluyendo las reglas de negocio nuevas o modificadas. 

Si bien las respuestas eran satisfactorias por parte del modelo se pudo notar un leve retraso en la respuesta del mismo, esto se puede deber al uso del modelo (gpt-4o-mini), a la vectorizacion de los archivos o por la forma en que consumimos el modelo (apikey). Como trabajo futuro el objetivo es que comprenda mas tipos de archivos y mejorar los tiempos de respuesta del modelo, asi podra dar mejor experiencia al usuario final

## Conclusión

En este documento se presentó un sistema de generación automática de código basado en reglas de negocio, aprovechando la potencia de RAG para asegurar que la salida del modelo esté respaldada por documentos formales de la organización. Los resultados preliminares evidencian el potencial de esta solución para acelerar los procesos de desarrollo de software en sectores donde abundan las reglas empresariales (como el financiero), ya que automatiza tareas repetitivas de codificación (por ejemplo, escribir validaciones o cálculos según políticas establecidas) y mantiene la coherencia con las reglas corporativas. En esencia, el enfoque permite que desarrolladores y analistas confíen en que cualquier código generado a partir de la documentación oficial refleje fielmente las intenciones y restricciones definidas por el negocio. A futuro, este tipo de sistemas podría integrarse en entornos de desarrollo para funcionar como asistentes inteligentes que garanticen el cumplimiento de normativas internas, disminuyendo errores humanos y liberando tiempo para que los equipos se enfoquen en tareas de mayor nivel.

## Integrantes
* Juan Diego Becerra Peña
* Daniel Esteban Ramos Jimenez
