function x = trustregion(fun, grad, hessian, x0, tr0, trmax, tol, maxiter)

eta1 = 0.25;
eta2 = 0.75;

gamma1 = 0.25;
gamma2 = 2;

x = x0;
tr = tr0;

for loop = 1:maxiter
    
    f = fun(x);
    dfdx = grad(x);
    dfdx2 = hessian(x);
    dx = subproblem(x, dfdx, dfdx2, tr, tol, maxiter);
    
    xnew = x + dx;
    fnew = fun(xnew);
    eps = max(abs(dfdx));
    
    if eps < tol
        break;
    end
    
    eta = extendtrustregion(dx, f, fnew, dfdx, dfdx2);
    if eta >= eta1
        x = xnew;
        if eta >= eta2
            tr = min([gamma2*tr, trmax]);
        end
    else
        tr = gamma1*tr;
    end
    
end

end

function dx = subproblem(x, dfdx, dfdx2, tr, tol, maxiter)

N = size(x,1);
lambda = 0.1;
I = eye(N);
iter = 0;

for loop = 1:maxiter

    A = dfdx2 + lambda*I;
    R = chol(A); % Cholesky decomposition.
    p = bicg(A, -dfdx);
    q = bicg(R', p);
    %p = - A\dfdx;
    %q = R'\p;
    
    dlambda = (norm(p)/norm(q))^2*(norm(p) - tr)/tr;
    lambda = lambda + dlambda;
    
    if (abs(dlambda) < tol)
        break;
    end
    
end

dx = p;

end

function eta = extendtrustregion(dx, fold, fnew, dfdx, dfdx2)

eta = (fold - fnew)/(-dfdx'*dx - 0.5*dx'*dfdx2*dx);

end