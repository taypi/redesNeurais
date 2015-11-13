function [yEst] = execQ1(x, w1, w2, bias)

    for i=1:size(x)(2)
        %% Camada oculta
        %--Potencial de ativação
        uN = w1(:,1)*1 + w1(:,2)*x(i);
      
        %--Sinal de saída do neuronio [Função de ativação sigmóide]
        yN(:,1) = 2/(1 + exp(-2*uN(:,1))) - 1;
      
        %% Camada de saída
        %--Potencial de ativação
        uNs = bias;
        uNs += sum(w2*yN);
      
        %--Sinal de saída do neurônio [Função de ativação linear]
        yEst(i) = uNs;

    endfor

end