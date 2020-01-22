# GMM
GMM(Gaussian Mixture Model) with EM(Expectation Maximization) algorithm

Gaussian Mixture of Linear Regression Models(#GMLRM)
- 혼합분포
    $$p(t | \bold{\theta})=\sum_{k=1}^{K} \pi_{k} \mathcal{N}\left(t | \mathbf{w}_{k}^{\mathrm{T}} \mathbf{\phi}, \beta^{-1}\right)$$
    - $$K$$ : Number of Model $$(k= 1,2,3 \cdots)$$
    - $$t$$ : 단일 출력
    
- log likelihood function
    $$\ln p(\mathbf{t} | \bold{\theta})=\sum_{n=1}^{N} \ln \left(\sum_{k=1}^{K} \pi_{k} \mathcal{N}\left(t_{n} | \mathbf{w}_{k}^{\mathrm{T}} \bold{\phi}_{n}, \beta^{-1}\right)\right)$$


    - $$M$$ : hiper parameter
    
    - $$\bold{w_k} =\left[\begin{array}{c} {w_{k0}}\\ {w_{k1}} \\ {\vdots} \\ {w_{k M-1}}  \end{array}\right]$$  : $$[M \times 1] matrix$$


    - $$\mathbf{\phi_n} =\left[\begin{array}{c} {\phi_{0}(\bold{x}_n)}\\ {\phi_{1}(\bold{x}_n)} \\ {\vdots} \\ {\phi_{M-1}(\bold{x}_n)}  \end{array}\right]$$  : $$[M \times 1] matrix$$


    - $$\mathbf{t}=\left(t_{1}, \ldots, t_{N}\right)^{\mathrm{T}}$$


- latent variable (잠재변수)
    - 잠재변수를 넣은 log-likelihood function
    $$\ln p(\mathbf{t}, \mathbf{Z} | \bold{\theta})=\sum_{n=1}^{N} \sum_{k=1}^{K} z_{n k} \ln \left\{\pi_{k} \mathcal{N}\left(t_{n} | \mathbf{w}_{k}^{\mathrm{T}} \mathbf{\phi}_{n}, \beta^{-1}\right)\right\}$$
    
- EM algorithm for maximize log-likelihood function
    - E-step : evaluate the posterior probabilities  $$p\left(k | \bold{\phi}_{n}, \bold{\theta}^{\text {old }}\right)$$ or responsibility $$\gamma_{nk}$$ 
    $$\gamma_{n k}=\mathbb{E}\left[z_{n k}\right]=p\left(k | \bold{\phi}_{n}, \bold{\theta}^{\mathrm{old}}\right)=\frac{\pi_{k} \mathcal{N}\left(t_{n} | \mathbf{w}_{k}^{\mathrm{T}} \bold{\phi}_{n}, \beta^{-1}\right)}{\sum_{j} \pi_{j} \mathcal{N}\left(t_{n} | \mathbf{w}_{j}^{\mathrm{T}} \bold{\phi}_{n}, \beta^{-1}\right)}$$


    - M-step : 
        - evaluate prior :
            - $$\pi_{k}=\frac{1}{N} \sum_{n=1}^{N} \gamma_{n k}$$


        - evaluate weight parameter :
            - $$\mathbf{w}_{k}=\left(\mathbf{\Phi}^{\mathrm{T}} \mathbf{R}_{k} \mathbf{\Phi}\right)^{-1} \mathbf{\Phi}^{\mathrm{T}} \mathbf{R}_{k} \mathbf{t}$$


                - $$\bold{\Phi}=\left[\begin{array}{c} {\phi_{1}}^{T} \\ {\phi_{2}}^{T} \\ {\vdots} \\ {\phi_{N}}^{T}  \end{array}\right]=\left[\begin{array}{cccc} {\phi_{0}(\mathbf{x}_1)} & {\phi_{1}(\mathbf{x}_1)} & {\cdots} & {\phi_{M-1}(\mathbf{x}_1)} \\ {\phi_{0}(\mathbf{x}_2)} & {\phi_{1}(\mathbf{x}_2)} & {\cdots} & {\phi_{M-1}(\mathbf{x}_2)}  \\ {\vdots} & {\vdots} & {\ddots} & {\vdots} \\ {\phi_{0}(\mathbf{x}_N)} & {\phi_{1}(\mathbf{x}_N)} & {\cdots} & {\phi_{M-1}(\mathbf{x}_N)}  \end{array}\right]$$$$[N \times M] matrix$$


                - $$\bold{R}_k = diag(\gamma_{nk}) =\left[\begin{array}{cccc} {\gamma_{1 k}} & {0} & {\cdots} & {0} \\ {0} & {\gamma_{2 k}} & {\cdots} & {0} \\ {\vdots} & {\vdots} & {\ddots} & {\vdots} \\ {0} & {0} & {\cdots} & {\gamma_{N k}} \end{array}\right]$$$$[N \times N] matrix$$


                - $$\beta$$
