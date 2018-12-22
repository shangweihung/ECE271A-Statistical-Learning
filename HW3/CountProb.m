function Pxy = CountProb(x, mu, sigma, prior, c)

Pxy = 0;    
for i = 1 : c
    Pxy = Pxy + prior(i) * mvnpdf(x, mu(i, :), diag(sigma(i,:)));
end

end