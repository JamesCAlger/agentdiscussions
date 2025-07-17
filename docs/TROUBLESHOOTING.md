# Troubleshooting Guide

This guide helps you diagnose and resolve common issues when using the Agentic Conversation System.

## Table of Contents

- [Quick Diagnostics](#quick-diagnostics)
- [Configuration Issues](#configuration-issues)
- [API and Connection Issues](#api-and-connection-issues)
- [Context Window Issues](#context-window-issues)
- [Performance Issues](#performance-issues)
- [Logging and Output Issues](#logging-and-output-issues)
- [Installation Issues](#installation-issues)
- [Error Reference](#error-reference)
- [Debug Mode](#debug-mode)
- [Getting Help](#getting-help)

## Quick Diagnostics

### System Health Check

Run these commands to quickly diagnose common issues:

```bash
# Check system information and configuration status
agentic-conversation info

# Validate your configuration file
agentic-conversation validate config.yaml --detailed

# Test with verbose logging
agentic-conversation run --config config.yaml --verbose --log-level DEBUG
```

### Environment Check

Verify your environment setup:

```bash
# Check Python version (requires 3.8+)
python --version

# Check if package is installed correctly
python -c "import agentic_conversation; print(agentic_conversation.__version__)"

# Check environment variables
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY
echo $LOG_LEVEL
echo $LOG_DIR
```

## Configuration Issues

### Error: "Configuration file not found"

**Symptoms:**
```
Error: Configuration file not found: config.yaml
```

**Causes:**
- File path is incorrect
- File doesn't exist
- Permission issues

**Solutions:**
```bash
# Check if file exists
ls -la config.yaml

# Use absolute path
agentic-conversation run --config /full/path/to/config.yaml

# Check current directory
pwd

# Create configuration from template
agentic-conversation create-config basic config.yaml
```

### Error: "Missing required field 'model_name'"

**Symptoms:**
```
Error: Missing required field 'model_name' in model configuration
```

**Causes:**
- Incomplete configuration file
- Typo in field name
- Missing required sections

**Solutions:**
```yaml
# Ensure model section has required fields
model:
  model_name: "gpt-4"  # This field is required
  temperature: 0.7
  max_tokens: 2000
```

**Validation:**
```bash
agentic-conversation validate config.yaml --detailed
```

### Error: "Agent A and Agent B must have different names"

**Symptoms:**
```
Error: Agent A and Agent B must have different names
```

**Causes:**
- Both agents have identical names
- Copy-paste error in configuration

**Solutions:**
```yaml
agents:
  agent_a:
    name: "Research Assistant"  # Make names unique
    system_prompt: "..."
  agent_b:
    name: "Analysis Expert"     # Different from agent_a
    system_prompt: "..."
```

### Error: "Temperature must be between 0.0 and 2.0"

**Symptoms:**
```
Error: Model: Temperature must be between 0.0 and 2.0
```

**Causes:**
- Invalid temperature value
- Typo in configuration

**Solutions:**
```yaml
model:
  model_name: "gpt-4"
  temperature: 0.7  # Must be between 0.0 and 2.0
```

### Error: "Failed to parse YAML configuration"

**Symptoms:**
```
Error: Failed to parse YAML configuration: ...
```

**Causes:**
- Invalid YAML syntax
- Indentation errors
- Special characters not properly escaped

**Solutions:**
```bash
# Validate YAML syntax online or with tools
python -c "import yaml; yaml.safe_load(open('config.yaml'))"

# Check indentation (use spaces, not tabs)
# Ensure consistent indentation levels
# Quote strings with special characters
```

**Common YAML Issues:**
```yaml
# ❌ Wrong - inconsistent indentation
agents:
  agent_a:
   name: "Agent"  # Wrong indentation
    system_prompt: "..."

# ✅ Correct - consistent indentation
agents:
  agent_a:
    name: "Agent"
    system_prompt: "..."

# ❌ Wrong - unquoted special characters
initial_prompt: Let's discuss AI: the future

# ✅ Correct - quoted strings with special characters
initial_prompt: "Let's discuss AI: the future"
```

## API and Connection Issues

### Error: "API key not found" or Authentication Errors

**Symptoms:**
```
Error: OpenAI API key not found
Error: Unauthorized: Invalid API key
Error: Authentication failed
```

**Causes:**
- Missing API key environment variable
- Invalid or expired API key
- Incorrect API key format

**Solutions:**
```bash
# Set OpenAI API key
export OPENAI_API_KEY="sk-your-actual-api-key-here"

# Set Anthropic API key
export ANTHROPIC_API_KEY="your-anthropic-api-key-here"

# Verify API key is set
echo $OPENAI_API_KEY

# Test API key validity
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/models
```

**Persistent Environment Variables:**
```bash
# Add to ~/.bashrc or ~/.zshrc
echo 'export OPENAI_API_KEY="your-key-here"' >> ~/.bashrc
source ~/.bashrc
```

### Error: "Connection timeout" or "Network error"

**Symptoms:**
```
Error: Request timed out
Error: Connection failed
Error: Network unreachable
```

**Causes:**
- Network connectivity issues
- Firewall blocking requests
- API service downtime
- Timeout settings too low

**Solutions:**
```bash
# Test internet connectivity
ping google.com

# Test API endpoint connectivity
curl -I https://api.openai.com/v1/models

# Increase timeout in configuration
```

```yaml
conversation:
  turn_timeout: 60.0  # Increase from default 30.0
```

**Corporate Network Issues:**
```bash
# Check proxy settings
echo $HTTP_PROXY
echo $HTTPS_PROXY

# Configure proxy if needed
export HTTP_PROXY="http://proxy.company.com:8080"
export HTTPS_PROXY="http://proxy.company.com:8080"
```

### Error: "Rate limit exceeded"

**Symptoms:**
```
Error: Rate limit exceeded
Error: Too many requests
```

**Causes:**
- API rate limits exceeded
- Too many concurrent requests
- Insufficient API quota

**Solutions:**
```bash
# Check API usage in provider dashboard
# Wait before retrying
# Reduce conversation frequency

# The system includes automatic retry logic
# Check logs for retry attempts
```

**Configuration Adjustments:**
```yaml
conversation:
  turn_timeout: 45.0  # Allow more time for rate-limited requests
```

### Error: "Model not found" or "Invalid model"

**Symptoms:**
```
Error: Model 'gpt-5' not found
Error: Invalid model specified
```

**Causes:**
- Typo in model name
- Model not available in your region
- Model deprecated or renamed

**Solutions:**
```yaml
# Use correct model names
model:
  model_name: "gpt-4"        # ✅ Correct
  # model_name: "gpt-5"     # ❌ Doesn't exist
  # model_name: "GPT-4"     # ❌ Wrong case
```

**Valid Model Names:**
- OpenAI: `"gpt-4"`, `"gpt-4-turbo"`, `"gpt-3.5-turbo"`
- Anthropic: `"claude-3-opus"`, `"claude-3-sonnet"`, `"claude-3-haiku"`

## Context Window Issues

### Error: "Context window exceeded"

**Symptoms:**
```
Error: Context window size exceeded
Warning: Context utilization at 95%
```

**Causes:**
- Conversation too long for context window
- Context window size set too high
- Large system prompts

**Solutions:**
```yaml
conversation:
  context_window_size: 6000    # Reduce from 8000
  context_window_strategy: "sliding"  # Use sliding window
  max_turns: 15               # Reduce conversation length
```

**Model-Specific Limits:**
```yaml
# GPT-3.5-Turbo (16K context)
conversation:
  context_window_size: 12000

# GPT-4 (8K context)
conversation:
  context_window_size: 6000

# GPT-4-Turbo (128K context)
conversation:
  context_window_size: 100000

# Claude-3 (200K context)
conversation:
  context_window_size: 150000
```

### Warning: "Context window approaching capacity"

**Symptoms:**
```
Warning: Context window utilization at 85%
```

**Causes:**
- Normal operation near context limits
- Inefficient context management

**Solutions:**
- Monitor context utilization in logs
- Adjust context window size if needed
- Use sliding window strategy

**Monitoring:**
```bash
# Analyze context utilization
agentic-conversation analyze ./logs --format detailed
```

## Performance Issues

### Slow Response Times

**Symptoms:**
- Long delays between agent responses
- Timeouts occurring frequently

**Causes:**
- Network latency
- Complex prompts
- Model processing time
- Rate limiting

**Solutions:**
```yaml
# Increase timeout
conversation:
  turn_timeout: 60.0

# Optimize model settings
model:
  max_tokens: 1500    # Reduce from 2000
  temperature: 0.5    # Lower temperature can be faster
```

**Performance Monitoring:**
```bash
# Check response times in logs
agentic-conversation analyze ./logs --format detailed

# Run with performance logging
agentic-conversation run --config config.yaml --verbose
```

### High Token Usage

**Symptoms:**
- Unexpectedly high API costs
- Rapid token consumption

**Causes:**
- Large context windows
- Long system prompts
- Many conversation turns

**Solutions:**
```yaml
# Optimize token usage
model:
  max_tokens: 1000    # Reduce response length

conversation:
  max_turns: 10       # Limit conversation length
  context_window_size: 4000  # Reduce context size
```

**Token Monitoring:**
```bash
# Analyze token usage
agentic-conversation analyze ./logs --format summary
```

### Memory Usage Issues

**Symptoms:**
- System running out of memory
- Slow performance over time

**Causes:**
- Large conversation histories
- Memory leaks
- Insufficient system resources

**Solutions:**
```bash
# Monitor memory usage
top -p $(pgrep -f agentic-conversation)

# Reduce memory usage
```

```yaml
conversation:
  max_turns: 20       # Limit conversation length
  context_window_size: 6000  # Reduce context size

logging:
  save_conversation_history: false  # Reduce memory usage
```

## Logging and Output Issues

### Error: "Permission denied" for log directory

**Symptoms:**
```
Error: Permission denied: ./logs
Error: Cannot create directory ./logs
```

**Causes:**
- Insufficient permissions
- Directory doesn't exist
- Disk space issues

**Solutions:**
```bash
# Create directory with proper permissions
mkdir -p ./logs
chmod 755 ./logs

# Use different directory
export LOG_DIR="/tmp/agentic-logs"

# Check disk space
df -h
```

### No log files created

**Symptoms:**
- Conversation runs but no log files appear
- Empty log directory

**Causes:**
- Logging disabled in configuration
- Permission issues
- Incorrect log directory

**Solutions:**
```yaml
logging:
  save_conversation_history: true  # Enable logging
  save_telemetry: true
  export_formats: ["json"]         # Specify formats
  output_directory: "./logs"       # Check directory
```

**Verification:**
```bash
# Check if logs are being created
ls -la ./logs/

# Check log directory permissions
ls -ld ./logs/
```

### Garbled or incomplete output

**Symptoms:**
- Truncated log files
- Corrupted JSON output
- Missing conversation data

**Causes:**
- Process interrupted
- Disk space issues
- File system problems

**Solutions:**
```bash
# Check disk space
df -h

# Check file system errors
fsck /dev/your-disk

# Ensure clean shutdown
# Use Ctrl+C to interrupt gracefully
```

## Installation Issues

### Import Error: "No module named 'agentic_conversation'"

**Symptoms:**
```python
ImportError: No module named 'agentic_conversation'
ModuleNotFoundError: No module named 'agentic_conversation'
```

**Causes:**
- Package not installed
- Wrong Python environment
- Installation failed

**Solutions:**
```bash
# Install package
pip install -e .

# Check installation
pip list | grep agentic

# Verify Python environment
which python
which pip

# Use virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

### Dependency Issues

**Symptoms:**
```
Error: No module named 'langchain'
Error: No module named 'langgraph'
```

**Causes:**
- Missing dependencies
- Version conflicts
- Incomplete installation

**Solutions:**
```bash
# Install with dependencies
pip install -e ".[dev]"

# Update dependencies
pip install --upgrade langchain langgraph

# Check dependency versions
pip list | grep lang

# Resolve conflicts
pip install --force-reinstall langchain
```

### Command not found: "agentic-conversation"

**Symptoms:**
```bash
agentic-conversation: command not found
```

**Causes:**
- Package not installed correctly
- PATH issues
- Wrong installation method

**Solutions:**
```bash
# Reinstall package
pip install -e .

# Check if command is available
which agentic-conversation

# Run directly with Python
python -m agentic_conversation.main --help

# Check PATH
echo $PATH
```

## Error Reference

### Configuration Errors

| Error Message | Cause | Solution |
|---------------|-------|----------|
| `Configuration file not found` | File path incorrect | Check file path and permissions |
| `Failed to parse YAML` | Invalid YAML syntax | Validate YAML syntax |
| `Missing required field` | Incomplete configuration | Add missing required fields |
| `Invalid log level` | Wrong log level value | Use: DEBUG, INFO, WARNING, ERROR, CRITICAL |
| `Temperature must be between 0.0 and 2.0` | Invalid temperature | Set temperature in valid range |

### API Errors

| Error Message | Cause | Solution |
|---------------|-------|----------|
| `API key not found` | Missing environment variable | Set API key environment variable |
| `Unauthorized` | Invalid API key | Check API key validity |
| `Rate limit exceeded` | Too many requests | Wait and retry, check rate limits |
| `Model not found` | Invalid model name | Use correct model identifier |
| `Connection timeout` | Network issues | Check connectivity, increase timeout |

### Runtime Errors

| Error Message | Cause | Solution |
|---------------|-------|----------|
| `Context window exceeded` | Context too large | Reduce context size or use sliding window |
| `Agent timeout` | Response took too long | Increase turn_timeout |
| `Permission denied` | File permission issues | Check directory permissions |
| `Circuit breaker open` | Too many failures | Wait for circuit breaker to reset |

## Debug Mode

### Enable Comprehensive Debugging

```bash
# Set debug environment
export LOG_LEVEL=DEBUG

# Run with maximum verbosity
agentic-conversation run \
  --config config.yaml \
  --verbose \
  --log-level DEBUG

# Save debug output
agentic-conversation run \
  --config config.yaml \
  --verbose \
  --log-level DEBUG \
  2>&1 | tee debug.log
```

### Debug Configuration

```yaml
logging:
  log_level: "DEBUG"
  output_directory: "./debug-logs"
  real_time_display: true
  export_formats: ["json", "txt"]
  save_conversation_history: true
  save_telemetry: true
```

### Analyzing Debug Output

```bash
# Search for specific errors
grep -i error debug.log

# Check API calls
grep -i "api" debug.log

# Monitor token usage
grep -i "token" debug.log

# Check context management
grep -i "context" debug.log
```

## Getting Help

### Before Asking for Help

1. **Check this troubleshooting guide** for your specific issue
2. **Run diagnostics** using the commands provided above
3. **Collect relevant information**:
   - Error messages (full text)
   - Configuration file (remove API keys)
   - System information (`agentic-conversation info`)
   - Debug logs (if applicable)

### Information to Include

When reporting issues, include:

```bash
# System information
agentic-conversation info

# Configuration validation
agentic-conversation validate config.yaml --detailed

# Error logs (remove sensitive information)
tail -50 debug.log

# Environment information
python --version
pip list | grep -E "(langchain|langgraph|agentic)"
```

### Common Solutions Summary

| Problem Category | Quick Fix |
|------------------|-----------|
| Configuration | Run `agentic-conversation validate config.yaml` |
| API Issues | Check API keys and network connectivity |
| Context Issues | Reduce context_window_size or max_turns |
| Performance | Increase timeouts, reduce token limits |
| Installation | Reinstall with `pip install -e ".[dev]"` |
| Permissions | Check file/directory permissions |

### Self-Help Resources

1. **Configuration Reference**: See `docs/CONFIGURATION.md`
2. **API Documentation**: See `docs/API.md`
3. **Example Configurations**: Use `agentic-conversation create-config`
4. **Log Analysis**: Use `agentic-conversation analyze ./logs`

### Community Support

1. **Search existing issues** in the repository
2. **Create detailed issue reports** with all relevant information
3. **Use appropriate issue templates** when available
4. **Be patient and respectful** when asking for help

This troubleshooting guide covers the most common issues encountered when using the Agentic Conversation System. For additional help, refer to the other documentation files or create an issue in the repository.