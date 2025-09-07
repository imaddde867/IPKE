#!/bin/bash
# EXPLAINIUM - CSC OpenShift Deployment Script

echo "🚀 EXPLAINIUM CSC Deployment"
echo "============================"

# Check if oc command exists
if ! command -v oc &> /dev/null; then
    echo "❌ Error: 'oc' command not found. Please install OpenShift CLI."
    echo "📋 Download from: https://docs.openshift.com/container-platform/latest/cli_reference/openshift_cli/getting-started-cli.html"
    exit 1
fi

# Check if we're logged in
if ! oc whoami &> /dev/null; then
    echo "❌ Error: Not logged in to OpenShift. Please run: oc login <csc-rahti-url>"
    exit 1
fi

echo "✅ Logged in as: $(oc whoami)"
echo "📁 Project: $(oc project -q)"

# Deploy the application
echo "🚀 Deploying EXPLAINIUM to CSC..."
oc create -f k8s/deploy.yaml

# Wait for deployment
echo "⏳ Waiting for deployment to complete..."
oc rollout status dc/explainium-app

# Get the route URL
echo "🌐 Getting application URL..."
ROUTE_URL=$(oc get route explainium-route -o jsonpath='{.spec.host}')
echo "✅ Application deployed!"
echo "🔗 Access your app at: https://$ROUTE_URL"

# Show status
echo "📊 Deployment Status:"
oc get pods
echo ""
echo "🔧 Useful commands:"
echo "  - View logs: oc logs dc/explainium-app"
echo "  - Scale app: oc scale dc/explainium-app --replicas=2"
echo "  - Delete app: oc delete -f k8s/deploy.yaml"
