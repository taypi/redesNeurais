function [vX, vYd] = genSample(n, step)
    
    cont = 1;
    for i = 1:n
        vYd(i) = gs(cont);
        vX(i) = cont;
        cont=cont+step;
    endfor

endfunction