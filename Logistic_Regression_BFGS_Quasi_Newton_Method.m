data = load("C:\Users\medra\Documents\Rice University\" + ...
    "531 Convex Optimization\HW2\LogReg_data_2020.mat")

A = data.A0;
b = data.b(:);
[m , n] = size(A);

epsilon = 1e-10;
alpha = 1e-1;
beta = 0.8;
t = 1;
max_iteration = 20;

% Use initial point (w0, c0) = 0
w = zeros(n, 1);
c = 0;
Bw = eye(n);
Bc = 1;

gradient_norms_list = [];

w_new = w;
c_new = c;

while true
    [gw, gc] = gradient_logreg(w_new, c_new, m, A, b);
    gradient_norm = sqrt(gw' * gw + norm(gc)^2);

    gradient_norms_list(end+1) = gradient_norm;

    disp(gradient_norm);

    if gradient_norm < epsilon
        break;
    end
    
    pw = -(Bw \ gw);
    pc = -(Bc \ gc);

    tk = amijo(w_new, c_new, m, gw, gc, pw, pc, A, b, beta, alpha, t, max_iteration);
    
    w_new = w_new + tk * pw;
    c_new = c_new + tk * pc;

    sw = w_new - w;
    sc = c_new - c;

    [new_gw, new_gc] = gradient_logreg(w_new, c_new, m, A, b);

    yw = new_gw - gw;
    yc = new_gc - gc;

    Bw = Bw - (Bw*sw*(sw')*Bw)/(sw'*Bw*sw) + (yw*yw')/(yw'*sw);
    Bc = yc / sc;

    w = w_new;
    c = c_new;
end

% Optimization function
function f = logreg(w, c, m, A, b)
    z = b .* (A * w + c);
    f = (1/m) * sum(log(1+exp(-z)));
end

% Compute the gradient
function [gw, gc] = gradient_logreg(w, c, m, A, b)
    z = b .* ( A * w + c);
    p = 1 ./ (1 + exp(-z));
    one_minus_p = 1 - p;

    gw = -((A .* b)' * one_minus_p) / m;
    gc = -(b' * one_minus_p) / m;
end

function tk = amijo(w, c, m, gw, gc, pw, pc, A, b, beta, alpha, t, max_iteration)
    % Backtracking line search
    tk = t;

    for k = 1:max_iteration
        w_new = w + tk * pw;
        c_new = c + tk * pc;

        % Armijo condition
        lhs = logreg(w_new, c_new, m, A, b);
        rhs = logreg(w, c, m, A, b) + alpha * tk * (gw' * pw + gc * pc);

        if lhs >= rhs
            tk = beta * tk;
        else
            return;
        end
    end
end

figure;
semilogy(0:length(gradient_norms_list)-1, gradient_norms_list, ...
    'y.-','MarkerFaceColor','r','LineWidth',1.5);
xlabel('Iteration');
ylabel('||\nabla f||_2');
title('Euclidean norm of the gradient vs iteration number');
grid off;