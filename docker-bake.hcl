target "app" {
    context = "."
    dockerfile = "Dockerfile"
    tags = ["ghcr.io/badrmoh/mlops-practitoner-project-2/legrag:latest"]
    platforms = ["linux/amd64", "linux/arm64"]
}
