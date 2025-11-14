## Authentication

The API server supports token-based authentication for both API endpoints and the admin panel.

### Setup Authentication

1. **Generate API tokens:**
   ```bash
   python scripts/generate_tokens.py
   ```

2. **Configure server with tokens:**
   ```bash
   export API_AUTH_TOKENS="token1,token2,token3"
   export CEREBRAS_API_KEY='your-cerebras-api-key'
   python src/api_server.py
   ```

3. **Configure client:**
   ```bash
   export CEREBRAS_PROXY_API_TOKEN="token1"
   ```

**Note:** If `API_AUTH_TOKENS` is not set, the server runs in development mode without authentication.

## Admin Configuration Panel

Access the web-based admin panel to manage configuration without restarting the server:

### Features

- 📊 **Real-time Statistics**: Monitor queue status, worker count, and active tasks
- ⚙️ **Worker Management**: Adjust worker concurrency dynamically
- 🤖 **Model Configuration**: Add, remove, and configure models on-the-fly
- 🌐 **Global Settings**: Modify retry logic, timeouts, and load monitoring
- 💾 **Persistent Changes**: Configuration changes are saved to YAML file
- 🔄 **Live Reload**: Reload configuration from file without restart

### Access Admin Panel

1. **Start the API server:**
   ```bash
   cd src
   python api_server.py
   ```

2. **Open your browser:**
   ```
   http://localhost:8000/admin
   ```

3. **Login with your API token**

### Configuration Changes

All changes made through the admin panel:
- ✅ Take effect immediately
- ✅ Are saved to `config/inference-config.yaml`
- ✅ Persist across server restarts
- ✅ Don't require worker restart

### Supported Configuration

- **Worker Concurrency**: Adjust total concurrent workers
- **Model Settings**: Configure per-model concurrency and generation parameters
- **Retry Logic**: Tune retry delays and maximum attempts
- **Load Monitoring**: Adjust TTFT thresholds and request timeouts
- **Generation Parameters**: Temperature, top_p, max_tokens, streaming