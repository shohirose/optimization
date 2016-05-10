%% NEWTON METHOD FOR MINIMIZATION PROBLEM

% This function find the local minimum by using Newton method with
% line search.

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
    dx = bicg(H, -df);
    %dx = -H\df;
    
    s = linesearch(fun, x, dx, tol, maxiter);
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

function s = linesearch(fun, x, dx, maxiter)

s = 0;
perturbs = 1e-6;

for loop = 1:maxiter
    
    xm = x + (s - perturbs)*dx;
    x0 = x + s*dx;
    xp = x + (s + perturbs)*dx;
    
    fm = fun(xm);
    f0 = fun(x0);
    fp = fun(xp);
    
    dfds = (fp - fm)/perturbs;
    dfds2 = (fp - 2*f0 + fm)/perturbs^2;
    ds = -dfds/dfds2;
    
    snew = s + ds;
    
    if snew > 1
        snew = 1;
    end
    
    fnew = fun(x + snew*dx);
    
    if  fnew > f0
        break;
    elseif snew == 1
        s = 1;
        break;
    else
        s = snew;
    end
    
end

if loop >= maxiter
    error('The iteration in linesearch() did not converge.');
end

end