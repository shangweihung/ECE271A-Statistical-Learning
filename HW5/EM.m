function [mu_final,var_final,pi_final]=EM(dimen,muprev,varprev,piprev,c,data)

        picur = piprev;
        mucur = muprev;
        varcur = varprev;

        [Numofdata, column] = size(data);
        
        % Initialize the h matrix
        h = zeros(Numofdata, c);
        
        
        for times = 1 : 100
            

            % E-step: create h matrix
            % hij= G(x_i,mu_j,sig_j)*pi_j  / Sigma k=1 to C (G(x_i,mu_k,sig_k)*pi_k)
            for i = 1 : Numofdata
                for j = 1:c
                    h(i, j) = piprev(j)* mvnpdf(data(i, :), muprev(j, :), diag(varprev(j, :))) ;
                end
                h(i, :) = h(i, :) / sum(h(i, :));
            end

             % M-step: calculate new parameters
            for j = 1:c
                % sum of Sigma of i hij
                sumh = sum(h(:, j));
                % mu(n+1)= Sigma of i hij *x_i /Sigma of i hij
                mucur(j, :) = h(:, j)'*data(:, :)./sumh;
                % pi(n+1)= Sigma of i hij /n
                picur(j) = sumh / Numofdata;
                
               sum_1=0;
               %sigma^2(n+1)=Sigma of i hij (x_i-m_j)^2 /Sigma of i hij
               for i=1: size(data,1)
                   sum_1=sum_1+h(i,j)*(data(i,:)-mucur(j,:)).^2;
               end
               varcur(j, :) = sum_1 / sumh;
               varcur(varcur < 1e-4) = 1e-4;
            end
            % another terminating criterion when the change is too little
            if all(abs(mucur - muprev)./abs(muprev) < 1e-4)
                break;
            end
            muprev = mucur;
            piprev = picur;
            varprev = varcur;
        end

        pi_final = piprev;
        mu_final = muprev;
        var_final = varprev;
   
end