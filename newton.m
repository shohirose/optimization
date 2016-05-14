%% NEWTON METHOD FOR MINIMIZATION PROBLEM

% This function find the local minimum by using Newton method with
% line search. The algorithm is taken from 
% J. Nocedal and S.J. Wright, 1999, Numrical Optimization, Springer series
% in operations research.

% Author: Sho Hirose, 2016/5/10

% +-----------------------------------------------------------------------+
% |                        Definition of Variables                        |
% +-----------------------------------------------------------------------+
% x       : the variable of f(x), vector.
% fun     : the function to be minimized, f(x).
% grad    : gradient vector of f(x).
% hessian : hessian matrix of f(x).
% x0      : initial x.
% tol     : convergence tolerance.
% maxiter : maximum iteration.

function x = newton(fun, grad, hessian, x0, tol, maxiter)

x = x0;

for loop = 1:maxiter
    
    df = grad(x);
    H = hessian(x);
    %dx = bicg(H, -df);
    dx = -H\df;
    
    s = linesearch(fun, grad, x, dx);
    x = x + s*dx;
    
    eps = max(abs(df));
    
    if eps < tol
        break;
    end
    
end

if loop >= maxiter
    error('The iteration in newton() did not converge.');
end

end

%% Line search function

function steplength = linesearch(fun, grad, x, dx)

phi = @(a) fun(x + a*dx);
dphi = @(a) grad(x + a*dx);

s = 1;

% Strong Wolfe parameter.
c1 = 1e-4;
c2 = 0.9;

f0 = phi(0);
g0 = dphi(0)'*dx;

sold = 0;
fold = f0;
gold = g0;

maxiter = 100;
for loop = 1:maxiter
    
    f1 = phi(s);
    
    if ((f1 > (f0 + c1*s*g0)) || ((f1 > fold) && (loop > 1)))
        steplength = zoom(phi, dphi, dx, sold, s);
        break;
    end
    
    g1 = dphi(s)'*dx;
    
    if abs(g1) <= -c2*g0
        steplength = s;
        break;
    elseif g1'*dx >= 0
        steplength = zoom(phi, dphi, dx, s, sold);
        break;
    end
    
    snew = updates(sold, s, fold, f1, gold, g1);
    sold = s;
    fold = f1;
    gold = g1;
    s = snew;
    
end

if loop >= maxiter
    error('The iteration in linesearch() did not converge.');
end

end

%% Zoom function

function steplength = zoom(fun, grad, dx, s1, s2)

slo = 0;
shi = 1;
flo = fun(slo);
fhi = fun(shi);
f0 = fun(0);
g0 = grad(0)'*dx;
c1 = 1e-4;
c2 = 0.9;

for i = 1:20
    snew = bisection(s1, s2);
    fnew = fun(snew);
    if (fnew > (f0 + c1*snew*g0)) || (fnew >= flo)
        shi = snew;
        fhi = fun(shi);
    else
        gnew = grad(snew)'*dx;
        if abs(gnew) <= -c2*g0
            steplength = snew;
            break;
        end
        if gnew*(shi - slo) >= 0
            shi = slo;
            fhi = fun(shi);
        end
        slo = snew;
        flo = fun(slo);
    end
end

end

%% Update step length by using bisection method

function snew = bisection(s1, s2)
snew = 0.5*(s1 + s2);
end

%% Update step length by using cubic interpolation
% 
% $$\alpha_{i+1} = \alpha_i - (\alpha_i - \alpha_{i-1}) \left[ \frac{\phi'(\alpha_i) + d_2 - d_1}{\phi'(\alpha_i) - \phi'(\alpha_{i-1}) + 2d_2} \right]$$
% $$d_1 = \phi'(\alpha_{i-1}) + \phi'(\alpha_i) - 3\frac{\phi(\alpha_{i-1}) - \phi(\alpha_i)}{\alpha_{i-1} - \alpha_i}$$
% $$d_2 = \left[ d_1^2 - \phi'(\alpha_{i-1})\phi'(\alpha_i) \right]^{1/2}$$
%
% s1 : $\alpha_{i-1}$
% s2 : $\alpha_i$
% f1 : $\phi(\alpha_{i-1})$
% f2 : $\phi(\alpha_i)$
% g1 : $\phi'(\alpha_{i-1})$
% g2 : $\phi'(\alpha_i)$

function snew = updates(s1, s2, f1, f2, g1, g2)
d1 = g1 + g2 - 3*(f1 - f2)/(s1 - s2);
d2 = sqrt(d1^2 - g1*g2);
if imag(d2) ~= 0
    d2 = real(d2);
end
snew = s2 - (s2 - s1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2));
end