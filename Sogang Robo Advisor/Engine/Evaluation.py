import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, kurtosis, skew
import base64
from io import BytesIO

plt.rcParams['axes.unicode_minus'] = False


class visualization:
    """
    The visualization class provides methods for analyzing and visualizing financial data.
    It focuses on creating cumulative return plots, comparing performance metrics between 
    strategy and benchmark, and exporting results in JSON format.

    Attributes:
        _data (pd.DataFrame): The input data containing 'Model' and 'Benchmark' columns.

    Methods:
        cumlativeReturns(json_filename="cumulative_returns.json"):
            Computes and visualizes cumulative returns for both strategy and benchmark, 
            and exports the results as JSON.
        
        compareFrame():
            Computes and returns a comprehensive set of performance metrics, including:
            cumulative returns, CAGR, Sharpe ratio, Sortino ratio, maximum drawdown, 
            volatility, and others for strategy and benchmark.
    """

    def __init__(self, dataframe):
        self._data = dataframe

    def cumlativeReturns(self, json_filename="cumulative_returns.json"):
        df = self._data

        # df["Model"] = (1 + df["Model"]).cumprod() - 1
        # df["Benchmark"] = (1 + df["Benchmark"]).cumprod() - 1

        df = df.dropna()
        dates = pd.to_datetime(df.index)
        strategy_returns = [x * 100 for x in df['Model']]
        benchmark_returns = [x * 100 for x in df['Benchmark']]

        plt.figure(figsize=(10, 6))
        plt.plot(dates, strategy_returns, label='Strategy', color='blue', linewidth=2)
        plt.plot(dates, benchmark_returns, label='Benchmark', color='gold', linewidth=2)

        plt.title('Cumulative Returns vs SPY', fontsize=14)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Cumulative Returns (%)', fontsize=12)

        plt.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        graph_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()
        plt.close()

        json_data = {
            "dates": dates.strftime("%Y-%m-%d").tolist(),
            "strategy_returns": strategy_returns,
            "benchmark_returns": benchmark_returns,
            # "graph_image": graph_base64  # Base64로 인코딩된 그래프 이미지
        }
        return json_data

    def compareFrame(self):
        df = self._data
        df = df.dropna()
        strategy_returns = df['Model']
        benchmark_returns = df['Benchmark']

        def cumulative_return(returns):
            if returns.empty or returns.isnull().all():
                return np.nan
            return (1 + returns).prod() - 1

        def cagr(returns, periods_per_year=252):
            total_return = (1 + returns).prod()
            n_years = len(returns) / periods_per_year
            if n_years == 0:
                return np.nan
            return total_return ** (1 / n_years) - 1

        def sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
            excess_returns = returns - risk_free_rate / periods_per_year
            if returns.std() == 0:
                return np.nan
            return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()

        def sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
            excess_returns = returns - risk_free_rate / periods_per_year
            downside_std = returns[returns < 0].std()
            if downside_std == 0 or np.isnan(downside_std):
                return np.nan
            return np.sqrt(periods_per_year) * excess_returns.mean() / downside_std

        def max_drawdown(returns):
            cumulative = (1 + returns).cumprod()
            drawdown = cumulative / cumulative.cummax() - 1
            if returns.empty:
                return np.nan
            return drawdown.min()

        def annual_volatility(returns, periods_per_year=252):
            if returns.std() == 0:
                return 0
            return returns.std() * np.sqrt(periods_per_year)

        def calmar_ratio(returns, periods_per_year=252):
            max_dd = max_drawdown(returns)
            if max_dd == 0:
                return np.nan
            return cagr(returns, periods_per_year) / abs(max_drawdown(returns))

        def skewness(returns):
            if len(returns) < 3:
                return np.nan
            return skew(returns)

        def kurtosis_calc(returns):
            if len(returns) < 3:
                return np.nan
            return kurtosis(returns)

        def calculate_expected_returns_d(returns):
            expected_daily = np.mean(returns) * 100
            return expected_daily

        def calculate_expected_returns_m(returns):
            returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
            returns = np.clip(returns, -1, 1)

            try:
                mean_return = np.mean(returns)
                if mean_return < -1 or mean_return > 1:
                    raise ValueError(f"Mean return out of range: {mean_return}")

                expected_monthly = ((1 + mean_return) ** 21 - 1) * 100
                return expected_monthly
            except OverflowError as e:
                print("OverflowError in monthly return calculation:", e)
                return None
            except ValueError as e:
                print("ValueError in monthly return calculation:", e)
                return None

        def calculate_expected_returns_y(returns):
            returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
            returns = np.clip(returns, -1, 1)

            try:
                mean_return = np.mean(returns)
                if mean_return < -1 or mean_return > 1:
                    raise ValueError(f"Mean return out of range: {mean_return}")

                expected_yearly = ((1 + mean_return) ** 252 - 1) * 100
                return expected_yearly
            except OverflowError as e:
                print("OverflowError in yearly return calculation:", e)
                return None
            except ValueError as e:
                print("ValueError in yearly return calculation:", e)
                return None

        def calculate_kelly_criterion(returns):
            mean_return = np.mean(returns)
            variance = np.var(returns)
            if variance == 0:
                return np.nan
            kelly_criterion = (mean_return / variance) * 100
            return kelly_criterion

        def calculate_risk_of_ruin():
            return 0.0

        def calculate_var(returns, confidence_level=0.95):
            z_score = norm.ppf(1 - confidence_level)
            daily_volatility = np.std(returns)
            if daily_volatility == 0:
                return np.nan
            daily_var = z_score * daily_volatility * 100
            return daily_var

        def calculate_cvar(returns, confidence_level=0.95):
            z_score = norm.ppf(1 - confidence_level)
            daily_volatility = np.std(returns)
            if daily_volatility == 0:
                return np.nan
            cvar = z_score * daily_volatility - (norm.pdf(z_score) / (1 - confidence_level)) * daily_volatility
            return cvar * 100

        metrics = {
            "Metric": [
                "Cumulative Return",
                "CAGR%",
                "Sharpe",
                "Sortino",
                "Max Drawdown",
                "Volatility (ann.)",
                "Calmar",
                "Skew",
                "Kurtosis",
                "Expected Daily %",
                "Expected Monthly %",
                "Expected Yearly %",
                "Kelly Criterion",
                "Risk of Ruin",
                "Daily Value-at-Risk (VaR)",
                "Expected Shortfall (CVaR)"],

            "Strategy": [
                cumulative_return(strategy_returns) * 100,
                cagr(strategy_returns) * 100,
                sharpe_ratio(strategy_returns),
                sortino_ratio(strategy_returns),
                max_drawdown(strategy_returns) * 100,
                annual_volatility(strategy_returns) * 100,
                calmar_ratio(strategy_returns),
                skewness(strategy_returns),
                kurtosis_calc(strategy_returns),
                calculate_expected_returns_d(strategy_returns),
                calculate_expected_returns_m(strategy_returns),
                calculate_expected_returns_y(strategy_returns),
                calculate_kelly_criterion(strategy_returns),
                calculate_risk_of_ruin(),
                calculate_var(strategy_returns),
                calculate_cvar(strategy_returns)],

            "Benchmark": [
                cumulative_return(benchmark_returns) * 100,
                cagr(benchmark_returns) * 100,
                sharpe_ratio(benchmark_returns),
                sortino_ratio(benchmark_returns),
                max_drawdown(benchmark_returns) * 100,
                annual_volatility(benchmark_returns) * 100,
                calmar_ratio(benchmark_returns),
                skewness(benchmark_returns),
                kurtosis_calc(benchmark_returns),
                calculate_expected_returns_d(benchmark_returns),
                calculate_expected_returns_m(benchmark_returns),
                calculate_expected_returns_y(benchmark_returns),
                calculate_kelly_criterion(benchmark_returns),
                calculate_risk_of_ruin(),
                calculate_var(benchmark_returns),
                calculate_cvar(benchmark_returns)]
        }

        """     
        JSON / PkL 파일 저장 함수 및 코드 -> 사용시 주석 및 코멘트 해제
        def save_metrics_to_files(metrics, json_filename="metrics.json", pkl_filename="metrics.pkl"):
            # PKL 파일로 저장
            metrics.to_pickle(pkl_filename)
            print(f"Metrics saved as PKL: {pkl_filename}")
            
            # JSON 파일로 저장
            metrics.to_json(json_filename, orient="records", indent=4)
            print(f"Metrics saved as JSON: {json_filename}")
        
        df_metrics = pd.DataFrame(metrics)  
        save_metrics_to_files(df_metrics)
        """
        return metrics


class indexVisualization:
    """
    The indexVisualization class focuses on visualizing portfolio allocations. It provides 
    methods for processing and visualizing stock allocation percentages in a structured format.

    Attributes:
        _data (dict): A dictionary containing stock symbols as keys and their respective 
                      allocation percentages as values.

    Methods:
        Visualization():
            Processes the input allocation data, sorts it, and returns a JSON representation.
    """

    def __init__(self, datadict):
        self._data = datadict

    def Visualization(self):

        data = self._data

        def indexMetrics(data):
            stocks_metrics = {}
            for stock, per in data.items():
                try:
                    stocks_metrics[stock] = round(per * 100, 2)
                except ValueError:
                    # print(f"Invalid format in data: {data[i]}")
                    continue

            stocks_metrics_sorted = dict(list(sorted(stocks_metrics.items(), key=lambda item: item[1], reverse=True)))
            return stocks_metrics_sorted

        """
        # 개별 종목 그래프 생성 및 이미지 저장
        def generateGraphImage(stock, percentage, maxvalue):
     
            # 그래프 생성
            plt.figure(figsize=(6, 1))
            plt.barh([stock], [percentage], color='skyblue', edgecolor='black')  # 가로 막대 그래프
            plt.xlim(0, maxvalue + 3)  # X축 범위를 전체 비율 최대값 +3으로 설정
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            plt.show()  # 그래프 화면 출력
            
            # 그래프를 PNG 이미지로 저장 및 Base64로 변환
            buffer = BytesIO()  # 이미지 데이터를 임시 저장할 버퍼 생성
            plt.savefig(buffer, format='png')  # 그래프를 버퍼에 저장
            buffer.seek(0)  # 버퍼의 시작점으로 이동
            graph_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')  # Base64 인코딩
            buffer.close()  # 버퍼 닫기
            plt.close()  # 그래프 닫기
            
            return graph_base64  # Base64로 인코딩된 그래프 이미지 반환
        
        # 종목별 데이터를 처리하고, 그래프 이미지를 포함한 딕셔너리 생성
        def generateStockDict(data,json_filename="stock_barplot.json"):
            
            stock_dict = {}
            maxvalue = max(list(data.values()))  # 비율의 최대값 계산 (X축 범위 설정에 사용)
            for stock, percentage in data.items():
                graph_image = generateGraphImage(stock, percentage, maxvalue)  # 종목별 그래프 생성
                stock_dict[stock] = {
                    "percentage": percentage,  # 각 종목의 비율
                    #"graph": graph_image       # Base64로 인코딩된 그래프 이미지
                }
            
    
            json 파일 저장 code
            with open(json_filename, "w") as json_file:
                json.dump(stock_dict, json_file, indent=4)
    
            print(f"Stock data and graphs saved to {json_filename}")
            json_result = json.dumps(stock_dict,indent = 4)
            return json_result# 최종 데이터 반환
        """
        metrics = indexMetrics(data)
        return metrics


def show(data_dic, data_df):
    index = indexVisualization(data_dic)
    index_bar_dict = index.Visualization()

    graph = visualization(data_df)
    graph_dict = graph.cumlativeReturns()

    metrics = visualization(data_df)
    metrics_dict = metrics.compareFrame()

    json_data = {
        "index_bar": index_bar_dict,  # 첫 번째 화면
        "graph": graph_dict,  # 두 번째 화면
        "metrics": metrics_dict  # 세 번째 화면
    }
    return json_data
