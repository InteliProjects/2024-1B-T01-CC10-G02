# Critérios de avaliação individual

## Nota N: Nota individual do aluno

A nota individual (N) do aluno é uma nota de 0 a 10 única por Sprint e que será composta por:

1. (P) Participação nos eventos da Sprint (_Onboarding_, _Workshop_, _Plannings_, _Reviews-Retros_ e _Dailies_). 
2. (G) Contribuições individuais no repositório Git

O seu cálculo é feito pela fórmula:

$$N = 0.25\times{P} + 0.75\times{G}$$

**Exemplo:**

Se a nota P do aluno for 8.0 e a sua nota G for 9.0, sua nota individual será:

$$N = 0.25\times{8.0} + 0.75\times{9.0} = 8.75$$

## Nota P: Participação nos eventos da Sprint

A nota de participação nos eventos (P) é uma nota individual de 0 a 10.

Essa nota sofrerá decréscimos de acordo com a seguinte lista:

| Item | Ocorrência  | Efeito na nota |
|--|-------------|------------------|
|(a)| Falta em 1 evento da Sprint (_Onboarding, Workshop, Planning, Reviews-Retros_) | -2 pontos |
|(b)| Falta em 1 _daily_ da Sprint | -0.5 ponto |

Estes decréscimos na nota são acumulativos. 

**Exemplo:**

Suponha que o aluno faltou a 2 eventos (por exemplo, na _Planning_ e na _Review-Retro_) e a 4 _dailies_. 

Assim ele terá um decréscimo de 4 pontos pelo item (a) da tabela e um decréscimo de 2 pontos pelo item (b), e sua nota de participação P será:

$$P = 10.0 - 4.0 - 2.0 = 4.0$$


## Nota G: Contribuições individuais no repositório Git

A nota de contribuições individuais no repositório Git (G) é uma nota individual de 0 a 10. Essa nota será determinada em 3 etapas.

### 1<sup>a</sup> Etapa da nota G: Determinação da faixa da nota

A faixa da nota G será determinada pelo atendimento dos seguintes critérios:

| Item | Critério |
|--|--|
|(a)|Fez _commits_ com contribuições significativas na Sprint.|
|(b)|Os _branches_ estão vinculados a _cards_ do planejamento e a _pull requests_.|
|(c)|Contribuiu em codificação e mais outro tipo de atividade.|
|(d)|Realizou os commits de forma distribuída ao longo do período da Sprint.|

> Observação: nas sprints em que não for solicitada a entrega de código, o item (c) levará em consideração se contribuiu em mais de um artefato.

Veja as [respostas a perguntas frequentes](faq.md) sobre os itens deste critério.

As faixas de nota são definidas da seguinte maneira:

| Critérios atendidos | Faixa da nota |
|--|--|
| atendeu a 1 único critério | 0 a 2.5 |
| atendeu a 2 critérios  | 2.6 a 5.0 |
| atendeu a 3 critérios  | 5.1 a 7.5 |
| atendeu aos 4 critérios  | 7.6 a 10.0 |

**Exemplo:**

Suponha que um aluno fez 7 _commits_ com contribuições significativas na Sprint atual que teve duração de 2 semanas (atendeu ao critério *a*), e que todos os seus _commits_ estão vinculados a _cards_ do planejamento e a _pull requests_ (atendeu ao critério *b*). Além disso, ele trabalhou no código do algoritmo de planejamento e na elaboração do documento de modelagem (atendeu ao critério *c*). No entanto, ele fez todos os _commits_ na véspera da entrega (não atendeu ao critério *d*).

Assim, ele atendeu a 3 critérios e sua nota G ficará na faixa 5.1 a 7.5.

### 2<sup>a</sup> Etapa da nota G: Determinação da nota dentro da faixa

O valor exato da nota G dentro da faixa definida no passo anterior será feita através da checagem de alguns pontos qualitativos. O avaliador considerará como ponto de partida a nota intermediária dentro da faixa, e aplicará acréscimos ou decréscimos de 0.5 ponto ao fazer a verificação dos seguintes itens:

| Item | Tipo | Descrição |
|--|--|--|
|(a)|negativo (-)|Há _commits_ que não parecem fazer sentido (por exemplo, um _commit_ que insere algo que é imediatamente retirado no próximo _commit_).|
|(b)|negativo (-)|Há _commits_ com mensagens pouco explicativas (por exemplo, um _commit_ com a mensagem "Update arquivo.").|
|(b)|negativo (-)|Cards planejados sem comentários.|
|(c)|negativo (-)|Há _pull requests_ com mensagens pouco explicativas (por exemplo, um _pull request_ com a mensagem "Merge branch documentação.").|
|(d)|positivo (+)|Foi criado um _pull request_ e efetuado um _merge_ para cada funcionalidade.|
|(e)|positivo (+)|Nos _cards_ há evidências de que as contribuições foram discutidas dentro da equipe utilizando os comentários.|
|(f)|positivo (+)|Há evidências de que todas as contribuições foram feitas de maneira precisa, com redação e autoria próprias.|
|(f)|negativo (-)|A contribuição parece ser um simples *copy-paste* de um conteúdo ou código encontrado pronto.|

Note que esta tabela contém somente alguns exemplos. Cada linha desta tabela pode ser usado também no sentido inverso. Por exemplo, o item (b) diz que mensagens pouco explicativas conta de forma negativa; isso também significa que mensagens de _commit_ muito bem redigidas (que explicam bem a contribuição deste _commit_) contam de forma positiva.

Os acréscimos e decréscimos são realizados desde que a nota não saia da faixa determinada no passo anterior.

**Exemplo:**

Suponha que o aluno, no passo anterior, teve sua nota G definida dentro da faixa de 7.6 a 10.0.

O avaliador utiliza como ponto de partida a nota no meio desta faixa (8.8).

Ao fazer a verificação dos itens da 2<sup>a</sup> etapa, foi identificado que, apesar do aluno ter contribuído com diversos commits, alguns deles não parecem fazer sentido, pois consistiram na inclusão de arquivos que, logo no _commit_ seguinte, foram excluídos (item *a*). Além disso, um dos _commits_ realizados têm uma mensagem vazia (item *b*).

Esta verificação indica que poderia ser aplicado um desconto de 1.0 ponto na nota. No entanto, o avaliador pode decidir aplicar um desconto de 0.6 porque a violação do item (b) ocorreu em somente um dos _commits_, e todos os demais 4 _commits_ estão bem descritos.

Assim, a nota de contribuição no Git seria:

$$G = 8.8 - 0.6 = 8.2$$

### 3<sup>a</sup> Etapa da nota G: Entrevista (a critério do orientador)

O orientador pode convidar os alunos para entrevistas para verificar a familiaridade do aluno com os artefatos em que ele contribuiu. De acordo com o desempenho do aluno nesta entrevista, a nota G pode aumentar ou diminuir.

> Observação: Se o aluno, quando convidado, não comparecer à entrevista, sua nota G terá um desconto de 50%.

**Exemplo:**

Suponha que o aluno, após a segunda etapa, teve sua nota G definida como 8.2. No entanto, na entrevista, ele não conseguiu mostrar em seu computador a execução do software em que havia feito contribuições (o que é uma evidência de que provavelmente não conseguiu executar o software após ter feito as contribuíções). Neste caso, a sua nota terá uma redução de 20%:

$$G = {8.2}\times{(1 - 0.2)} = 6.56$$
