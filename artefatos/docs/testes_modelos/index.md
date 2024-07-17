# Testes de modelos CNN 

## SegNet

 Com o avanço das redes neurais convolucionais profundas (CNNs), surgiu uma demanda crescente por métodos eficientes e precisos de segmentação semântica. O SegNet é uma arquitetura de CNN projetada a partir da necessidade de mapear características de baixa resolução para resolução de entrada, mantendo informações detalhadas de borda para uma classificação precisa de pixels. O SegNet consiste em uma rede codificadora, inspirada na VGG16, e uma rede decodificadora correspondente, seguida por uma camada de classificação de pixel a pixel, como demostrado na imagem a seguir:

 ![Arquitetura SegNet](/docs/img/segNet.jpg)

 No processo de codificação, as características são extraídas por meio de convoluções e normalizadas em lote, seguidas por ativação ReLU e max-pooling para alcançar invariância à translação. No entanto, para preservar detalhes de borda, o SegNet armazena apenas os índices de max-pooling, em vez das características completas. O processo de decodificação envolve o upsampling dos mapas de características usando os índices de max-pooling armazenados, seguido por convolução e normalização em lote para reconstruir mapas de características densos. Finalmente, uma camada de classificação softmax produz probabilidades de classe para cada pixel, resultando na segmentação prevista.
 Comparado a outras arquiteturas semelhantes, como DeconvNet e U-Net, o SegNet se destaca por sua eficiência e uso inteligente de índices de max-pooling para upsampling, resultando em uma arquitetura leve e eficiente em termos de recursos computacionais e de memória.

Na entrega da sprint 2, nosso foco foi na implementação do modelo de classificação. No entanto, reconhecemos a importância de explorar outras abordagens em segmentação para enriquecer e aprimorar nosso modelo. Planejamos incorporar técnicas de transfer learning utilizando o SegNet para alcançar esse objetivo.

**Referências:**
- [Vijay Badrinarayanan, Alex Kendall, Roberto Cipolla. SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation](https://arxiv.org/abs/1511.00561)


## AlexNet

O AlexNet, arquitetura de CNN criada por Alex Krizhevsky, ganhou visibilidade ao vencer o desafio de reconhecimento de imagem ImageNet em 2012. Na época, a arquitetura se destacou por vencer com folga a competição com erro de 16%, enquanto o segundo colocado obteve um erro de aproximadamente 26% e, por isso, representou um marco no campo da visão computacional. Uma de suas principais características inovativas foi a utilização da função de ativação ReLU para introduzir não-linearidade no modelo, o seu uso permitiu treinamentos muito mais rápidos e custos muito baixos (por não envolver exponenciação) comparado a outras funções comumentes utilizados na época, como sigmoid e tanh. Outro aspecto importante e diferencial na arquitetura da rede foi o uso do Dropout para evitar o overfitting, técnica inovadora naquele período. Outra grande inovação que a AlexNet adotou foi o uso de GPU ao invés de CPU para o seu treinamento, algo que impulsionou e muito a velocidade do treino.

Sua estrutura é composta por 5 camadas convolucionais, 3 max pooling e 3 camadas totalmente conectadas, seguindo a estrutura da imagem abaixo:

![Arquitetura AlexNet](/docs/img/AlexNet.png)

**Referências:**
 - [Medium.com - A keras segnet implementation for building detection in the spacenet dataset](https://medium.com/@qucit/a-keras-segnet-implementation-for-building-detection-in-the-spacenet-dataset-edd23933f1f4)
 - [Wikipedia - AlexNet](https://en.wikipedia.org/wiki/AlexNet)
 - [Medium.com - AlexNet with TensorFlow](https://medium.com/swlh/alexnet-with-tensorflow-46f366559ce8)

## UNet

A UNet é uma rede neural convolucional que, inicialmente, foi idealizada para segmentação de imagens médicas. Seus diferenciais incluem o uso de skip connections, que preservam detalhes importantes durante a reconstrução da imagem segmentada. Além disso, a técnica de cropping reduz a perda de informações, garantindo uma contextualização precisa. Um destaque para a presença da UNet nesta documentação é sua capacidade de se obter bons resultados com conjuntos de dados pequenos, graças à sua habilidade de generalizar padrões aprendidos. Isso a torna uma escolha confiável em aplicações onde dados rotulados são escassos, como no trabalho desenvolvido no presente projeto que segmenta talhões produtivos em imagens de satélites. 

Sua arquitertura pode ser resumida por um caminho de contração e um caminho de expansão. O caminho de contração usa convoluções 3x3 seguidas de ReLU e max pooling 2x2 para reduzir a dimensionalidade. Cada passo duplica o número de canais de características. O caminho de expansão realiza upsampling, seguido por convoluções 2x2, concatenação com o caminho de contração correspondente e mais convoluções. Uma convolução final de 1x1 mapeia características para o número de classes desejado. A rede tem 23 camadas convolucionais. A arquiterura supracitada pode ser visualizada na imagem abaixo:

![Arquiterura UNet](/docs/img/U-Net-arquiteturajpg.jpg)

**Referências:**
- [Artigo UNet para segmentações de imagens médicas](https://paperswithcode.com/paper/u-net-convolutional-networks-for-biomedical)
- [Wikipedia - Unet](https://en.wikipedia.org/wiki/U-Net#:~:text=The%20network%20consists%20of%20a,and%20a%20max%20pooling%20operation.)




