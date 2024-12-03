from Engine.Tree import *
from Engine.BaseOptimizer import *
from Engine.Optimizer import *
from Engine.Assumption import *
from Engine.Pipeline import *
from Engine.Backtest import *
from Engine.DataReader import *
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


def build_investment_tree(codes: list, risk_level: int, df: pd.DataFrame) -> Tree:
    tree = Tree("Universe")

    if risk_level in [1, 2]:  # 안정추구형
        bounds = {
            5: (0, 0.15),
            4: (0, 0.4),
            3: (0, 0.2),
            2: (0, 1),
            1: (0, 1)
        }
    elif risk_level == 3:  # 위험중립형
        bounds = {
            5: (0, 0.2),
            4: (0, 0.8),
            3: (0, 1),
            2: (0, 1),
            1: (0, 1)
        }
    else:  # 적극투자형 (4, 5)
        bounds = {
            5: (0, 0.2),
            4: (0, 1),
            3: (0, 1),
            2: (0, 1),
            1: (0, 1)
        }

    row_code_dict = {}
    for _, row in df.iterrows():
        row_code_dict[row.iloc[0]] = str(row['종목 코드'])

    code_info = {}
    for code in codes:
        row = df[df['종목 코드'].astype(str) == str(code)].iloc[0]
        code_info[code] = {
            'level': row['level'] if not pd.isna(row['level']) else 0,
            'parent': row['부모'] if not pd.isna(row['부모']) else None,
            'risk': row['투자 가능 대상 여부']
        }

    sorted_codes = sorted(codes, key=lambda x: code_info[x]['level'], reverse=True)
    for code in sorted_codes:
        bound = bounds[code_info[code]['risk']]
        if code_info[code]['level'] != 2:
            if row_code_dict[code_info[code]['parent']] != code:
                if row_code_dict[code_info[code]['parent']] not in [node.name for node in tree.get_all_nodes()]:
                    tree.insert('Universe', row_code_dict[code_info[code]['parent']], weight_bounds=bound)
                    tree.insert(row_code_dict[code_info[code]['parent']], code, weight_bounds=bound)

                elif row_code_dict[code_info[code]['parent']] in [node.name for node in tree.get_all_nodes()]:
                    tree.insert(row_code_dict[code_info[code]['parent']], code, weight_bounds=bound)

            else:
                if code not in [node.name for node in tree.get_all_nodes()]:
                    tree.insert('Universe', code, weight_bounds=bound)
                    tree.insert(code, code, weight_bounds=bound)
                elif code not in [node.name for node in tree.get_all_nodes()]:
                    tree.insert(code, code, weight_bounds=bound)

        elif code_info[code]['level'] == 2:
            tree.insert(code, code, weight_bounds=bound)

    tree.draw()
    return tree


# 사용방법
# codes: 투자할 종목 리스트
# risk_level: 투자자의 위험등급 (1~5, 5: 고위험 투자자)
# investor_goal: 투자자 목표(1: 결혼자금 준비, 2: 노후자금 준비, 3: 장기수익 창출, 4: 목돈마련)
def main(codes, risk_level: int = 4, investor_goal: int = 1):
    file_path = './invest_universe.csv'
    universe = pd.read_csv(file_path, encoding='cp949')

    stock_dict = {}
    for _, row in universe.iterrows():
        code = str(row['종목 코드'])
        if len(code) < 6:
            code = '0' * (6 - len(code)) + code
        stock_dict[row['종목 설명']] = code

    universe['종목 코드'] = stock_dict.values()
    tree = build_investment_tree(codes, risk_level, universe)

    assets = tree.get_all_nodes_name()

    if '069500' not in assets: assets.append('069500')

    price_data = fetch_data_from_db(assets, './financial_data.db')

    assumption = AssetAssumption(returns_window=52, covariance_window=52)

    ## 투자자 목표 1: 결혼자금 준비
    if investor_goal == 1:
        steps = [
            ("SAA", dynamic_risk_optimizer),
            ("TAA", mean_variance_optimizer),
            ("AP", mean_variance_optimizer)
        ]

    ## 투자자 목표 2: 노후자금 준비
    elif investor_goal == 2:
        steps = [
            ("SAA", risk_parity_optimizer),
            ("TAA", goal_based_optimizer),
            ("AP", mean_variance_optimizer)
        ]

    ## 투자자 목표 3 장기 수익 창출
    elif investor_goal == 3:
        steps = [
            ("SAA", risk_parity_optimizer),
            ("TAA", mean_variance_optimizer),
            ("AP", mean_variance_optimizer)
        ]

    ## 투자자 목표 4 목돈 마련
    elif investor_goal == 4:
        steps = [
            ("SAA", dynamic_risk_optimizer),
            ("TAA", goal_based_optimizer),
            ("AP", mean_variance_optimizer)
        ]

    pipe = Pipeline(steps, tree, assumption)

    today = datetime.today().strftime("%Y-%m-%d")
    prev_date = (datetime.today() - timedelta(days=252 * 5)).strftime("%Y-%m-%d")
    rebalance_dates = pd.date_range(prev_date, today, freq='M')
    trading_days = price_data.index

    backtest = Backtest(pipe, price_data, rebalance_dates, trading_days)
    backtest.run_backtest()

    allocation = backtest.allocations[-1][-1]
    allocation = {name: allocation[code] for name, code in stock_dict.items() if code in allocation}
    # df = pd.DataFrame([{'Date': date, **allocations} for date, allocations in backtest.allocations])
    # df.to_csv('check4.csv')

    eval_metrix = backtest.evaluation(allocation)
    return eval_metrix


if __name__ == '__main__':
    main(codes=['069500', '139260', '161510', '273130', '439870', '251340', '114260'], risk_level=4, investor_goal=3)
