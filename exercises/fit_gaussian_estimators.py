from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    univar = UnivariateGaussian()
    X = np.random.normal(10, 1, 1000)
    univar.fit(X)
    print((univar.mu_, univar.var_))

    # Question 2 - Empirically showing sample mean is consistent
    lengths = []
    distances = []
    for l in range(10, 1001, 10):
        lengths.append(l)
        sample = X[:l]
        univar.fit(sample)
        distances.append(abs(univar.mu_ - 10))
    go.Figure([go.Scatter(x=lengths, y=distances, mode='markers+lines', name=r'$\distance$')],
              layout=go.Layout(title=r"$\text{Question - 2}$",
                               xaxis_title="$m\\text{ - number of samples}$",
                               yaxis_title="r$\\text{distance between mu and estimation for sample}$",
                               height=500,
                               width=700)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    univar.fit(X)
    X.sort()
    pdfRes = univar.pdf(X)

    go.Figure([go.Scatter(x=X, y=pdfRes, mode='markers+lines', name=r'$\empirical data$')],

              layout=go.Layout(title=r"$\text{Question - 3}$",
                               xaxis_title="$\\text{sample values}$",
                               yaxis_title="r$\\text{pdf result}$",
                               height=500,
                               width=700)
              ).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = [0, 0, 4, 0]
    cov = [[1, 0.2, 0, 0.5],
           [0.2, 2, 0, 0],
           [0, 0, 1, 0],
           [0.5, 0, 0, 1]]
    X = np.random.multivariate_normal(mu, cov, 1000)
    multivar = MultivariateGaussian()
    multivar.fit(X)
    print(multivar.mu_)
    print(multivar.cov_)

    #print(multivar.pdf(X))
    print(MultivariateGaussian.log_likelihood(np.array(mu), np.array(cov), X))

    #Question 5 - Likelihood evaluation
    space = np.linspace(-10, 10, 200)

    getLogLikelihood = lambda f: multivar.log_likelihood(
        np.array([f[0],0,f[1],0]),
        np.array(cov),
        X
    )

    cartProdf1f3 = np.transpose(np.array([np.repeat(space,len(space)),np.tile(space,len(space))]))
    mapping = np.array(list(map(getLogLikelihood, cartProdf1f3)))
    logLikelihood = mapping.reshape(200,200)

    fig = go.Figure(go.Heatmap(x=np.linspace(-10, 10, 200), y=np.linspace(-10, 10, 200), z=logLikelihood),
                    layout=go.Layout(title="Question 5 - Log Likelihood f1, f3", height=500, width=700))
    fig.update_xaxes(title_text="f3 values")
    fig.update_yaxes(title_text="f1 values")
    fig.show()



    # Question 6 - Maximum likelihood
    maxVal = np.max(mapping)
    i = np.argmax(mapping)
    f1,f3 = cartProdf1f3[i]
    print(f"Maximum Likelihood: {round(maxVal, 3)}, f1: {round(f1, 3)}, f3:{round(f3,3)}")


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
