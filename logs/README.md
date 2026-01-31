# Pakit Logs

This directory contains runtime logs for Pakit services.

## Log Files

- `pakit_api.log` - API server logs
- `pakit_p2p.log` - P2P network logs
- `pakit_ml.log` - ML model serving logs

## Log Rotation

Logs are automatically rotated:
- **Max size**: 10 MB per file
- **Retention**: 7 days
- **Compression**: Gzip for archived logs

## Development

Logs are excluded from git (see `.gitignore`).

To enable debug logging:

```bash
export PAKIT_LOG_LEVEL=DEBUG
python api_server.py
```

## Production

For production deployments, configure log aggregation:
- Prometheus/Grafana for metrics
- ELK stack for log analysis
- CloudWatch/Stackdriver for cloud deployments

See `docs/DEPLOYMENT_GUIDE.md` for details.
