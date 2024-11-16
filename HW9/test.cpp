#include <iostream>
#include <vector>
#include <queue>

using namespace std;

vector<int> bfs(int root, const vector<vector<int>>& tree) {
    int n = tree.size();
    vector<int> level(n, -1);  // Initialize levels with -1 to indicate unvisited nodes
    queue<int> q;

    level[root] = 0;  // Start with the root node at level 0
    q.push(root);

    while (!q.empty()) {
        int node = q.front();
        q.pop();

        for (int neighbor : tree[node]) {
            if (level[neighbor] == -1) {  // Check if unvisited
                level[neighbor] = level[node] + 1;
                q.push(neighbor);
            }
        }
    }
    
    return level;
}

long long powerMod(long long n, long long p) {
    long long result = 1;
    long long base = 2;

    while (n > 0) {
        // If n is odd, multiply the current result by base
        if (n % 2 == 1) {
            result = (result * base) % p;
        }
        // Square the base and reduce n by half
        base = (base * base) % p;
        n /= 2;
    }

    return result;
}

vector<long long> solve(int n, vector<vector<int>> &edges, int q, vector<vector<int>> &queries)
{
    vector<vector<int>> adj(n);
    for(int i=0;i<n-1;i++)
    {
        int u,v; cin >> u >> v;
        adj[u-1].push_back(v-1);
        adj[v-1].push_back(u-1);
    }
    vector<int> level = bfs(0,adj);
    vector<long long> ans(q);
    for(int i=0;i<q;i++)
    {
        int u = queries[i][0]-1, v =queries[i][1]-1;
        int d = level[u]+level[v];
        ans[i] = (powerMod(d-1, 1000000007));
    }
    return ans;
}

int main() {
    

    return 0;
}
