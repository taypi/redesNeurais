function [w1, w2, bias, erroTrein, erroValid] = backPropag(nNeu, nAmostras, n, epocaMax, erroMax, vX, vYd, w1, w2, bias)
 lastDistanceError = 9999999999999999;
    for epoca = 1:epocaMax
      for k = 1:nAmostras
        x1 = vX(k);
        Yd = vYd(k);
        uN = w1(:,1)*1 + w1(:,2)*x1;

        yN(:,1) = 2/(1 + exp(-2*uN(:,1))) - 1;

        uNs = bias;
        uNs += sum(w2*yN);

        y = uNs;

        e = Yd - y;
        E(k) = e;

        grad2 = e;  
        sumGrad1 = grad2 * sum(w2);
        
        dBias = n * grad2;
        bias += dBias;
        dw2 = n * grad2 * yN;
        w2 += dw2';

        sumGrad1 = grad2 * sum(w2);

        dFSigmoide1N(1,:) = (1 - (((2 / (1 + exp(-2*uN(:,1)))) -1).^2));
        grad1N = sumGrad1*dFSigmoide1N;

        for i=1:nNeu
            dw1(i,1) = n * grad1N(i);
            dw1(i,2) = n * grad1N(i) * x1;
        endfor
        w1 = w1 + dw1;

      endfor

      [xV, yR] = genSample(nAmostras, 0.4);
      yV = execQ1(xV, w1, w2, bias);
      eV = yR - yV;
      EmV = (sum(sum(eV.*eV)))/2;

      % Teste do erro médio mínimo.
      Em = (sum(sum(E.*E)))/2;
      
      % fprintf("erroTrein");
      
      if (Em < erroMax && abs(Em - EmV) > lastDistanceError)
        break;
      endif
      
      lastDistanceError = abs(Em - EmV);

      % Imprimindo valores do treinamento
      %fprintf('Epoca Treinam. = %.0f Erro Médio Quadratico = %g\n\n', epoca, Em);
      
      erroTrein(epoca) = Em;
      erroValid(epoca) = EmV;
    endfor

    % fprintf("Backpropagation");
   
end