Ao início da Sprint 3, foi desenvolvido um modelo de segmentação baseado na arquitetura UNet, e com o objetivo de melhorar sua performance e resultados o transfer learning foi implementado. Como o UNet é divido em Encoder e Decoder, com o primeiro permitindo uma melhor identificação de características das imagens inputadas reduzindo sua solução espacial, enquanto o segundo "reconstroi" a imagem, com base no que o encoder caracterizou, aumentando sua resolução espacial.

O modelo pre-treinado utilizado foi o MobileNetV2, e apenas o seu encoder foi acoplado ao UNet original. Os principais ganhos com isso são a melhora na convergência e desempenho, dado que com um modelo pré-treinado, a rede já começa com pesos relevantes, ao invés de 0, o que leva a melhoras não apenas no treinamento mas também no desempenho final. Além disso, como o modelo foi treinado em uma base muito grande, a sua capacidade de generalização é imensa, implicando em um aprendizado positivo na identificação do que são bordas e formas.

Antes da implementação do encoder do MobileNetV2, os resultados do UNet implementado eram os seguintes:

![Resultados sem Transfer Learning](/docs/img/fotos_sem_tf.png)
![Resultados sem Transfer Learning](/docs/img/perda_sem_tf.png)
![Resultados sem Transfer Learning](/docs/img/precisao_sem_tf.png)

O modelo que gerou esses resultados obtidos acima tem uma acurácia avaliada em 82%, possuindo um total de 403.777 parâmetros, todos sendo treináveis.

Ao implementar o transfer learning, os resultados foram esses:

![Resultados com Transfer Learning](/docs/img/fotos_com_tf.png)
![Resultados com Transfer Learning](/docs/img/perda_com_tf.png)
![Resultados com Transfer Learning](/docs/img/precisao_com_tf.png)

Com um total de 502.241, sendo 436.321 treináveis e 65.920 não-treináveis, o modelo atingiu uma acurácia de 85%, com um treinamento muito mais rápido que o modelo UNet anterior "cru".