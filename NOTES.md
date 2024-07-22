# NOTES

RUN CHASS TO FIGURE IT OUT WHY THE PARK IS NOT TRAINNING. Chass does work so the issue apears to be with the park model.

Current hypothesis is that the model is not learning because of the statedict that is beeing sent to
the hybodesolver and then the lines 53-60 are not working as intended.

Change all torch.float64 to torch.float32 in the code.
Change all torch.float64 to torch.float32 in the code.

TESE \_ TRF, TRUST CONTRUCT, ADAM

ARTIGO \_ TRF 5, 5 5

CHASS \_ 1, 5, 1 1, 5 5

PARK \_ 1, 5, 1 1, 5 5

EXEMPLOS \_ 1, 5, 1 1, 5 5

PODER USAR O SITE SEM TER CONTA

METHODOLOGY FALAR MAIS SOBRE OS CASOS DE ESTUDO
FALAR SOBRE COMO OS DADOS FORAM GERADOS

FAZER UM DIAGRAMA MAIS ELABORADO ICON DO SITE PARA A TOOL
TOOL PARA FRONT END NO DIAGRAMA
https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.omdena.com%2Fblog%2Fmachine-learning-examples&psig=AOvVaw0QmI6Okr_HH0CWehP-NJro&ust=1721397707819000&source=images&cd=vfe&opi=89978449&ved=0CBEQjRxqFwoTCNjw5azgsIcDFQAAAAAdAAAAABBY

POR NO DIAGRAMA DA TOOL UMA SINTEZE DO WORKFLOW. Figure 4.4 tese

FIND A WAY TO MAKE SURE THE FLASK APP WORKS ON THE SERVER.
MAKE SURE THE REACT APP WORKS ON THE SERVER.
Meaning, the server should be able to run the flask app and the react app at the same time. Allow the user to access the react app through the server. Allow different users to make trainning requests to the server, and all being run at the same time.

NEW SIMULATION TAB FOR PLOTS WITH DIFERENT TIME STEPS

NOTAS SOBRE DOCUMENTO DA TESE PROF PEDRO:
pg 1. O que é o Biopharma 4.0? - chapter 1 paragraph 3

pg 4. Não se entende bem a figura 1.2. Explicar melhor no texto (p.ex. HMOD) ou remeter
para a secção 3.1.3 onde é bem explicado - DONE

pg 8. Não se entende bem a figura 2.1. Explicar melhor no texto. - DONE

pg 9 e seguintes:
Nestas secções (2.3.1 e 2.3.2) renomeiam-se
os first-principles science-based models (mechanistic models) como paramétricos
os data-based ML models como não paramétrico
Explicitar e uniformizar estas designações

pg 16 Naõ se entende a primeira frase da secção3.1.3. Será de cortar (pelo menos aqui),
talvez recuperando-a para a secção em que se descreve a nova tool. - DONE

pg 17 Início da secção 3.2. Será conveniente dar um nome à ferramenta que está a ser
desenvolvida, para se poder nomear convenientemente - HYBpy como referido às vezes?

       Dúvidas:
       1. A ferramenta MATLAB usava ou não o standard HMOD.
       2. Que diferenças/vantagens de "hibridização" oferecia em relação ao SBML2HYB

pg 19 A secçao 4.1.1 sobre o hybdata Script deveria explicar onde entra este script na
figura 4.1
O script já existia na versão MATLAB?

      Genérico - Deveria ser explicado (eventualmente com um exemplo simples) o formato
      de um ficheiro HMOD.

pg 22 Não entendo as 3 modalidades de treino, direta, indireta e semidireta.
Será talvez problema meu?

      Parameter Setup - aqui pode fazer confusão os parametros de ML e a designação de
      paramétricos aos modelos mecanicistas (ver nota atrás da pg 9)

pg 23 Usar os nomes dos layers por extenso da primeira vez que são usados -tanhLayer, ...

pg 25 Completar secção 4.5 e explicar figura.
