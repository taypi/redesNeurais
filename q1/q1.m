function [erroTrein] = q1(nNeu, nAmostras)
    
    % 1) Obter o conjunto de treinamento.
    % 2) Associar a saída desejada para cada amostra.
    [vX, vYd] = genSample(nAmostras, 0.5);

    r = randperm(length(vYd));
    vYd = vYd(r);
    vX = vX(r);

    % 3) Iniciar os vetores de pesos.    
    for i=1:nNeu
        w2(i) = 2*rand-1;
        for j=1:2
            w1(i,j) = 2*rand-1;
        endfor
    endfor

    bias = 2*rand-1;

    % 4) Especificar a taxa de aprendizagem(n) , 
    % precisão requerida(Emin) 
    % e o número máximo de épocas (epocaMax).
    n = 0.005;
    epocaMax = 10000;
    erroMax = 0.01;

    %%Treinamento Backpropagation

    [w1, w2, bias, erroTrein, erroValid] = backPropag(nNeu, nAmostras, n, epocaMax, erroMax, vX, vYd, w1, w2, bias);

    %% Gerando gráfico do erro

    t = 1:1:size(erroTrein,2);
    plot(t, erroTrein(1,:), 'k.', t, erroValid, 'b.');
    grid;
    xlabel('epoca');
    ylabel('Erro Medio Quadratico - Treinamento');
    figure

    [x, yR] = genSample(nAmostras, 0.6);
    yE = execQ1(x, w1, w2, bias);

    plot(x, yR, 'k.', x, yE, 'r.');
    
    %%Teste MLP

    % fprintf('---------------TesteMLP-----------------');
    

    % for i=1:10
    %     in = rand*20;
    %     yEstimado = execQ1(in, w1, w2);
    %     yReal = gs(in);

    %     fprintf('\nValores de entrada: %i\n', in);
    %     fprintf('Valor estimado pela MLP: %f\n', yEstimado);
    %     fprintf('Valor real: %f\n', yReal);

    % endfor



endfunction