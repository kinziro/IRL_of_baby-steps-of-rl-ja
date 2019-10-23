'''
こちらのソースコードは下記リポジトリの[baby-steps-of-rl-ja/IRL/maxent.py]のソースコードにコメントを追記しております。

https://github.com/icoxfog417/baby-steps-of-rl-ja

'''


import numpy as np
from planner import PolicyIterationPlanner
from tqdm import tqdm


class MaxEntIRL():

    def __init__(self, env):
        self.env = env
        self.planner = PolicyIterationPlanner(env)

    def estimate(self, trajectories, epoch=20, learning_rate=0.01, gamma=0.9):
        # 特徴量行列 F
        # 16×16の単位行列
        state_features = np.vstack([self.env.state_to_feature(s)
                                   for s in self.env.states])
        # 1. パラメータθを初期化
        theta = np.random.uniform(size=state_features.shape[1])
        # 2. エキスパートデータから特徴ベクトルに変換
        teacher_features = self.calculate_expected_feature(trajectories)

        for e in tqdm(range(epoch)):
            # Estimate reward.
            # 3. 状態ごとの報酬関数 R(s) = θ・F
            rewards = state_features.dot(theta.T)

            # 現時点のパラメータによる報酬関数を設定
            self.planner.reward_func = lambda s: rewards[s]
            # 4. 現時点の報酬関数に対して、方策を計算
            self.planner.plan(gamma=gamma)

            # 5. 計算した方策で特徴ベクトルを取得
            features = self.expected_features_under_policy(
                                self.planner.policy, trajectories)

            # 6. 勾配を計算
            # μ_expert - μ(θ)
            update = teacher_features - features.dot(state_features)
            theta += learning_rate * update

        estimated = state_features.dot(theta.T)
        estimated = estimated.reshape(self.env.shape)
        return estimated

    def calculate_expected_feature(self, trajectories):
        '''
        エキスパートデータから特徴ベクトルを作成する関数

        :param trajectories: エキスパートの軌跡データ
        :return: 特徴ベクトル
        '''
        features = np.zeros(self.env.observation_space.n)
        for t in trajectories:
            for s in t:
                features[s] += 1

        features /= len(trajectories)
        return features

    def expected_features_under_policy(self, policy, trajectories):
        '''
        パラメータによる報酬関数から獲得される方策による特徴ベクトルの取得

        :param policy: 方策
        :param trajectories: エキスパート軌跡  軌跡数×状態のリスト
        :return: 各状態の発生確率 16次元のリスト
        '''
        # エキスパート軌跡の数, このコードではt_size=20
        t_size = len(trajectories)
        states = self.env.states
        # 状態遷移確率 軌跡の数×状態数
        transition_probs = np.zeros((t_size, len(states)))

        # 状態の発生確率(各状態の到達頻度)
        initial_state_probs = np.zeros(len(states))
        # 各軌跡データの初期状態を取得して、数をカウント
        # 今回の場合は初期状態は12の為、transition_probs[12]=20でそれ以外は0
        for t in trajectories:
            initial_state_probs[t[0]] += 1
        # 回数を頻度に変換するためにt_sizeで割る
        initial_state_probs /= t_size
        # 状態遷移確率の初期行に初期状態の配列を代入
        transition_probs[0] = initial_state_probs

        # 環境の状態遷移確率にしたがって,t_size-1回状態遷移を繰り返して状態の発生確率を計算
        # 疑問なのが、なぜ繰り返すステップ数がt_sizeで良いのか？
        for t in range(1, t_size):
            # 1ステップ前の状態prev_sが状態0~15の場合をそれぞれ計算
            # 1ステップ前の状態の発生確率と状態遷移確率を掛けて全て足すことで今ステップの状態発生確率を計算
            # μ_t(s') = P(s'|s, a) * Σμ_t-1(s)
            for prev_s in states:
                # 1ステップ前に状態prev_sにいる確率を計算
                prev_prob = transition_probs[t - 1][prev_s]
                # 1ステップ前の状態prev_sで行う行動を方策から決定
                a = self.planner.act(prev_s)
                # 状態遷移確率に従い、各状態への遷移確率を取得. probsは16次元のリスト.
                probs = self.env.transit_func(prev_s, a)
                # 1ステップ前の発生確率と状態遷移確率を掛ける
                for s in probs:
                    transition_probs[t][s] += prev_prob * probs[s]

        # t_sizeステップ分の発生確率を平均して、各状態の発生確率を計算
        total = np.mean(transition_probs, axis=0)
        return total


if __name__ == "__main__":
    def test_estimate():
        # 環境の設定
        from environment import GridWorldEnv
        env = GridWorldEnv(grid=[
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 0],
        ])

        # エキスパートデータの収集
        teacher = PolicyIterationPlanner(env)
        teacher.plan()
        trajectories = []
        print("Gather demonstrations of teacher.")
        for i in range(20):
            s = env.reset()
            done = False
            steps = [s]
            while not done:
                a = teacher.act(s)
                n_s, r, done, _ = env.step(a)
                steps.append(n_s)
                s = n_s
            trajectories.append(steps)

        # 逆強化学習を実行
        print("Estimate reward.")
        irl = MaxEntIRL(env)
        rewards = irl.estimate(trajectories, epoch=100)
        print(rewards)
        env.plot_on_grid(rewards)

    test_estimate()
