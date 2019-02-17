# Introducción ligera al Machine Learning

En este post quiero (intentar) plasmar lo poco que se sobre el mundo de la Inteligencia Artificial, Machine Learning (ML), Redes Neuronales y un largo etcétera que dominan las noticias de hoy, usando un lenguaje que puede pecar de simplista o soez, pero que busca despertar curiosidad en aquellos no iniciados en ésta área y que tampoco dominan la programación.

Siempre estaré abierto a las críticas, aunque quiero dejar claro que muchas veces simplificaré en exceso los tecnicismos, mi meta final es que alguien que no sepa nada, al final de leer este post y, pasado algún tiempo pueda llegar a decir "Mario, creo que te equivocas aquí..." ¡Bingo! Ahora podemos hablar más detalladamente.

Bien, quedando claro esto, a lo que nos concierne...


## Inteligencia Artifical

Inteligencia Artificial puede ser vista como una rama de la ciencia que busca crear mecanismos para 'imitar' la inteligencia humana. Aquí me quiero detener en un comentario muy personal, mucho se habla de futuros apocalípticos en el que las máquinas inteligentes dominan a los humanos, la verdad no puedo saber si ese escenario se dará algún día, pero de que vamos a conseguir algo semejante a la 'inteligencia humana' es más bien una pregunta del 'cuándo' que de '¿lo lograremos?'. Que se yo, ¿cien años? ¿mil años?...

## Machine Learning

Machine Learning (ML), Aprendizaje Automático (o Automatizado, desconozco la traducción apropiada al español) no es más que un conjunto de técnicas que buscan crear modelos a partir de datos (ya va, ya voy a explicar esto en cristiano). Es un área dentro de la Inteligencia Artificial, así como la Química Orgánica es parte de lo que es la Química como ciencia en general (o, si prefieren, como el Jazz está dentro de lo que es la Música...).

Ok, ¿Qué es ésto de 'técnicas que buscan crear modelos a partir de datos'?, vamos a ir parte por parte.

### Modelos
En nuestro caso, los modelos son formas de representar algo, por ejemplo, un modelo que sirva para predecir el clima (algo increíblemente complejo, pero no prestemos atención a ello), un modelo para predecir el valor de una casa, etc. Simplifiquémoslo aun más, un modelo es algo a lo que podemos, por ejemplo, hacerle una pregunta como '¿Lloverá mañana?' y sea capaz de respondernos 'si' o 'no'.

¿Por qué creamos modelos? pues porque necesitamos saber algo que desconocemos, nos ayuda a tomar decisiones. Ahora bien, es normal pensar que para que un modelo sea útil debe otorgar información valiosa, de nada nos sirve un modelo que responda de forma aleatoria 'si' o 'no'. ¿Cómo podemos hacer un modelo que sea capaz de brindarnos información confiable?.

Esa, estimados, es la verdadera pregunta.

El mundo que nos rodea es tan variado, que buscar un modelo que responda todas nuestras preguntas es virtualmente imposible, las preguntas más simples implican una complejidad asombrosa. Por ello creamos modelos especializados para ciertas preguntas de interés (como los ya mencionados modelos climáticos y modelos de precios de propiedades).

Resulta ser que, muchas veces, no conocemos una 'fórmula mágica' que englobe todas las variables para responder las preguntas de interés. ¿Podemos predecir el precio de una casa solo por su tamaño?, es normal pensar que mientras más grande mas costosa, pero, ¿acaso la ubicación no es importante?, que me dicen de la cantidad de habitaciones, ventanas, sótano, ático, año de construcción, materiales utilizados, ¿nuevas construcciones proyectadas que tiendan a aumentar o disminuir este precio?....... Y seguro quienes trabajan en bienes raíces podrán identificar varias cosas más, porque, además, no es solo la cantidad o presencia/ausencia de alguna variable, es también la relación propia entre cada una (una casa extremadamente grande puede valer poco si está mal ubicada, muy vieja y en condiciones internas deplorables).

Otro aspecto a tener en cuenta es que, como seres humanos, tenemos nuestras propias limitantes importantes para estos casos, sirva de ejemplo la cantidad de multiplicaciones que podemos hacer en una hora, pongamos un número absurdamente falso de 1 por segundo, lo que nos daría 3600 multiplicaciones por hora. Los modelos complejos requieren miles de millones de multiplicaciones... y no podemos contar con que toda la humanidad se prestará todo el tiempo a hacer multiplicaciones...

Ahí es donde entran nuestras amigas las computadoras, que son capaces de realizar este tipo de tareas mucho (MUCHO) más rápido que nosotros. Sin profundizar más, la conclusión es que los modelos deben ser aptos para ser usados con computadoras.

Ok, seguimos sin decir cómo podemos crear esos modelos. Pensemos en un momento en cómo nosotros solemos aprender. Estamos (así lo creo) diseñados naturalmente para descubrir 'patrones', es decir, cosas que, dadas ciertas condiciones, se repiten una y otra vez. Por ejemplo, sabemos que el clima tiene ciertos ciclos anuales (estaciones), esto nos permite adaptarnos a situaciones que aún no han sucedido (¿comprar una chaqueta?). Pero, ¿cómo sabemos cuándo llegará determinada estación?, esto lo descubrimos hace muchos años, al ver como ciertos eventos se repetían pasada una determinada cantidad de tiempo, aunque no sabíamos el por qué, usábamos esos datos históricos en nuestro favor para crear un modelo de las temporadas.

Otra cosa es que, a medida que 'vemos' más algo, somos más capaces de 'entenderlo', así sea de forma empírica.

Entonces, si tan solo pudiéramos lograr que las computadoras descubrieran patrones usando datos.......

### Datos

Desde hace bastante tiempo, nos dimos cuenta que la cantidad de datos necesarios para crear modelos complejos era abrumadora. De hecho, la disponibilidad (captura/registro) y facilidad de manejo de datos (poder computacional) es lo que ha creado el boom actual del ML. Ilustremos mejor esta afirmación, por ejemplo, las redes neuronales en su núcleo no son más complejas que la multiplicación matricial y las derivadas parciales de primer orden (a nivel matemático, 'sencillas').

Dejaré a un lado el 'poder computacional', porque a nuestros efectos, solo implica 'mejores computadoras'.

En la actualidad, quizás el activo más importante que poseen las grandes compañías de tecnología (Google, Facebook, Apple, Amazon, etc) es la data a la que tienen acceso. Tan así que son capaces de brindar servicios masivos gratis que consideramos primordiales hoy en día (correo electrónico por ejemplo). Todo por acceder a nuestros datos. Pensemos en los datos como mineral sin procesar y los modelos las plantas que los convierten en cosas útiles.

La captura de datos no solo se manifiesta en servicios de consumo masivo, con el llamado 'Internet de las Cosas' (IOT en inglés), logramos colocar sensores por todos lados, monitoreando miles de cosas al mismo tiempo, llenando bases de datos con información a la espera de ser procesada.

Ahora bien, ¿Cómo podemos usar los datos para crear modelos?, un aspecto importante a considerar es que estos modelos deben ser lo suficientemente flexibles para adaptarse a nuevos datos, al igual que nosotros que somos capaces de ver un patrón, pero en caso necesario podemos cambiar nuestra percepción en presencia de nuevas experiencias

### Modelos basados en Datos

