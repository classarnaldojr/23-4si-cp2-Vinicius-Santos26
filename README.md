## Questão: Pedra, papel e tesoura

Você deve fazer um programa que detecta o vencedor do jogo pedra, papel e tesooura. 

Trata-se de um disputa entre 2 jogadores.

* Jogador1 = mão posicionada a esquerda do video
* Jogador2 = mão posicionada a direita do video

A cada rodada os jogadores escolhem sua jogada entre as opções:

- Pedra
- Papel 
- Tesoura

O resultado de cada rodada pode ser `empate` ou `vencedor` onde:

* Empate: Acontece quando os dois jogadores escolhem a mesma opção de jogada
* Vencedor: Acontece com as seguintes combinações de jogadas:
    - Pedra vence de Tesoura
    - Papel vence de Pedra
    - Tesoura vence de Papel 

### Orientações

Você deve criar um arquivo python chamado`jogo.py`. 

Você deve obrigatoriamente usar o vídeo `pedra-papel-tesoura.mp4` que já está no diretório.


### O que você deve fazer:

Criar script em python que devolve um output visual de acordo com a rubrica. 


### Rubrica

O que é esperado para cada rubrica:

|Resultado| Conceito| 
|----------|--------|
|R0 – Script não executa | 0 |
|R1 – Identifica ao menos um jogador corretamente e produz saída visual demonstrando | 0,8 |
|R2 – Identifica corretamente os dois jogadores e produz saídas visuais diferentes demonstrando | 1,2 |
|R3 – Identifica ao menos uma jogada dos jogadores e identifica escrevendo no output visual (ex: “JOGADOR `x` = `<PEDRA, PAPEL, TESOURA>`”). | 1,5 |
|R4 – Identifica corretamente todas as jogadas dos jogadores e identifica escrevendo no output visual | 2 |
|R5 – Compara as jogadas e informa se foi empate ou que jogador que venceu | 2 |
|R6 – Calcula a pontuação de cada jogador e exibe a pontuação total do jogo | 1 |
|R7 - Código organizado em funções com comentários pertinentes|1,5|

Casos intermediários ou omissos da rubrica serão decididos pelo professor.

a nota final é composta pelo somatorio simples de cada  rubrica. Nota = R1+R2+R3+R4+R5+R6+R7