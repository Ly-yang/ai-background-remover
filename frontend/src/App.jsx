// frontend/src/App.jsx
import React, { useState, useCallback, useEffect } from 'react';
import { Upload, Download, Settings, Zap, Image as ImageIcon, Loader2, Check, X } from 'lucide-react';
import axios from 'axios';
import './App.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const App = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [processedUrl, setProcessedUrl] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [taskId, setTaskId] = useState(null);
  const [progress, setProgress] = useState(0);
  const [settings, setSettings] = useState({
    model: 'rmbg-2.0',
    mode: 'auto',
    edgeOptimization: 'smooth',
    outputFormat: 'png'
  });
  const [dragActive, setDragActive] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [error, setError] = useState(null);
  const [processingTime, setProcessingTime] = useState(0);

  // 模型配置
  const models = {
    'rmbg-2.0': { name: 'RMBG 2.0', description: '最新模型，效果最佳，推荐使用', speed: 4, quality: 5 },
    'u2net': { name: 'U²-Net', description: '经典模型，速度快，通用性好', speed: 5, quality: 4 },
    'birefnet': { name: 'BiRefNet', description: '高精度边缘处理，细节丰富', speed: 3, quality: 5 },
    'sam': { name: 'SAM', description: 'Meta分割模型，适合复杂场景', speed: 2, quality: 5 },
    'isnet-anime': { name: 'ISNet-Anime', description: '专为动漫图像优化', speed: 4, quality: 5 }
  };

  const modes = {
    'auto': '自动检测',
    'person': '人物',
    'object': '物体',
    'animal': '动物',
    'product': '商品'
  };

  const edgeOptions = {
    'smooth': '平滑边缘',
    'sharp': '锐利边缘',
    'feather': '羽化边缘',
    'none': '无处理'
  };

  const outputFormats = {
    'png': 'PNG (透明背景)',
    'jpg': 'JPG (白色背景)',
    'webp': 'WebP (高压缩)'
  };

  // 文件拖拽处理
  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  }, []);

  const handleFileSelect = (file) => {
    if (!file.type.startsWith('image/')) {
      setError('请选择有效的图片文件');
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      setError('文件大小不能超过10MB');
      return;
    }

    setSelectedFile(file);
    setError(null);
    setProcessedUrl(null);
    setTaskId(null);
    setProgress(0);

    // 创建预览
    const reader = new FileReader();
    reader.onload = (e) => setPreviewUrl(e.target.result);
    reader.readAsDataURL(file);
  };

  // 处理图片
  const processImage = async () => {
    if (!selectedFile) return;

    setIsProcessing(true);
    setError(null);
    setProgress(0);
    const startTime = Date.now();

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('model', settings.model);
      formData.append('mode', settings.mode);
      formData.append('edge_optimization', settings.edgeOptimization);
      formData.append('output_format', settings.outputFormat);

      // 上传并处理
      const response = await axios.post(`${API_BASE_URL}/api/v1/process`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const { task_id } = response.data;
      setTaskId(task_id);

      // 轮询检查处理状态
      const checkStatus = async () => {
        try {
          const statusResponse = await axios.get(`${API_BASE_URL}/api/v1/task/${task_id}`);
          const { status, result_url } = statusResponse.data;

          if (status === 'completed') {
            setProcessedUrl(`${API_BASE_URL}${result_url}`);
            setIsProcessing(false);
            setProgress(100);
            setProcessingTime((Date.now() - startTime) / 1000);
          } else if (status === 'failed') {
            setError('处理失败，请重试');
            setIsProcessing(false);
          } else {
            // 模拟进度增长
            setProgress(prevProgress => Math.min(prevProgress + 10, 90));
            setTimeout(checkStatus, 1000);
          }
        } catch (err) {
          setError('检查状态时出错');
          setIsProcessing(false);
        }
      };

      setTimeout(checkStatus, 1000);

    } catch (err) {
      setError(err.response?.data?.detail || '处理失败，请重试');
      setIsProcessing(false);
    }
  };

  // 下载结果
  const downloadResult = async () => {
    if (!taskId) return;

    try {
      const response = await axios.get(`${API_BASE_URL}/api/v1/result/${taskId}`, {
        responseType: 'blob',
      });

      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `background_removed_${Date.now()}.${settings.outputFormat}`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      setError('下载失败，请重试');
    }
  };

  // 重置状态
  const reset = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setProcessedUrl(null);
    setTaskId(null);
    setProgress(0);
    setError(null);
    setIsProcessing(false);
    setProcessingTime(0);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* 背景动效 */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-purple-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-blue-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob animation-delay-2000"></div>
        <div className="absolute top-40 left-40 w-80 h-80 bg-pink-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob animation-delay-4000"></div>
      </div>

      {/* 导航栏 */}
      <nav className="relative z-10 bg-black/20 backdrop-blur-md border-b border-white/10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-2">
              <Zap className="w-8 h-8 text-purple-400" />
              <h1 className="text-2xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                AI Background Remover
              </h1>
            </div>
            <button
              onClick={() => setShowSettings(!showSettings)}
              className="p-2 rounded-lg bg-white/10 hover:bg-white/20 transition-colors"
            >
              <Settings className="w-5 h-5 text-white" />
            </button>
          </div>
        </div>
      </nav>

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* 错误提示 */}
        {error && (
          <div className="mb-6 p-4 bg-red-500/20 border border-red-500/30 rounded-lg flex items-center space-x-2">
            <X className="w-5 h-5 text-red-400" />
            <span className="text-red-200">{error}</span>
            <button
              onClick={() => setError(null)}
              className="ml-auto p-1 hover:bg-red-500/20 rounded"
            >
              <X className="w-4 h-4 text-red-400" />
            </button>
          </div>
        )}

        {/* 设置面板 */}
        {showSettings && (
          <div className="mb-8 p-6 bg-white/10 backdrop-blur-md rounded-2xl border border-white/20">
            <h2 className="text-xl font-semibold text-white mb-4">处理设置</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {/* 模型选择 */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">AI模型</label>
                <select
                  value={settings.model}
                  onChange={(e) => setSettings({...settings, model: e.target.value})}
                  className="w-full p-3 bg-black/30 border border-white/20 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                >
                  {Object.entries(models).map(([key, model]) => (
                    <option key={key} value={key}>{model.name}</option>
                  ))}
                </select>
                <p className="text-xs text-gray-400 mt-1">{models[settings.model]?.description}</p>
              </div>

              {/* 处理模式 */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">处理模式</label>
                <select
                  value={settings.mode}
                  onChange={(e) => setSettings({...settings, mode: e.target.value})}
                  className="w-full p-3 bg-black/30 border border-white/20 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                >
                  {Object.entries(modes).map(([key, label]) => (
                    <option key={key} value={key}>{label}</option>
                  ))}
                </select>
              </div>

              {/* 边缘优化 */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">边缘优化</label>
                <select
                  value={settings.edgeOptimization}
                  onChange={(e) => setSettings({...settings, edgeOptimization: e.target.value})}
                  className="w-full p-3 bg-black/30 border border-white/20 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                >
                  {Object.entries(edgeOptions).map(([key, label]) => (
                    <option key={key} value={key}>{label}</option>
                  ))}
                </select>
              </div>

              {/* 输出格式 */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">输出格式</label>
                <select
                  value={settings.outputFormat}
                  onChange={(e) => setSettings({...settings, outputFormat: e.target.value})}
                  className="w-full p-3 bg-black/30 border border-white/20 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                >
                  {Object.entries(outputFormats).map(([key, label]) => (
                    <option key={key} value={key}>{label}</option>
                  ))}
                </select>
              </div>
            </div>
          </div>
        )}

        {/* 主要内容区域 */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* 上传区域 */}
          <div className="space-y-6">
            <div
              className={`relative p-8 border-2 border-dashed rounded-2xl transition-all duration-300 ${
                dragActive 
                  ? 'border-purple-400 bg-purple-500/10' 
                  : 'border-white/30 bg-white/5 hover:bg-white/10'
              }`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              <input
                type="file"
                accept="image/*"
                onChange={(e) => e.target.files[0] && handleFileSelect(e.target.files[0])}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              />
              
              <div className="text-center">
                <div className="w-16 h-16 mx-auto mb-4 bg-gradient-to-r from-purple-400 to-pink-400 rounded-full flex items-center justify-center">
                  <Upload className="w-8 h-8 text-white" />
                </div>
                <h3 className="text-xl font-semibold text-white mb-2">
                  {selectedFile ? '更换图片' : '上传图片'}
                </h3>
                <p className="text-gray-400 mb-4">
                  拖拽图片到这里，或点击选择文件
                </p>
                <p className="text-sm text-gray-500">
                  支持 JPG、PNG、WebP 格式，最大 10MB
                </p>
              </div>
            </div>

            {/* 原图预览 */}
            {previewUrl && (
              <div className="bg-white/10 backdrop-blur-md rounded-2xl p-6 border border-white/20">
                <h3 className="text-lg font-semibold text-white mb-4">原图预览</h3>
                <div className="relative aspect-video bg-black/30 rounded-lg overflow-hidden">
                  <img
                    src={previewUrl}
                    alt="原图"
                    className="w-full h-full object-contain"
                  />
                </div>
                <div className="mt-4 flex justify-between items-center">
                  <span className="text-sm text-gray-400">
                    {selectedFile?.name} ({(selectedFile?.size / 1024 / 1024).toFixed(2)} MB)
                  </span>
                  <button
                    onClick={reset}
                    className="px-4 py-2 bg-red-500/20 hover:bg-red-500/30 text-red-200 rounded-lg transition-colors"
                  >
                    重新选择
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* 结果区域 */}
          <div className="space-y-6">
            {/* 处理按钮 */}
            {selectedFile && !isProcessing && !processedUrl && (
              <button
                onClick={processImage}
                className="w-full py-4 bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white font-semibold rounded-2xl transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-xl"
              >
                <div className="flex items-center justify-center space-x-2">
                  <Zap className="w-5 h-5" />
                  <span>开始处理</span>
                </div>
              </button>
            )}

            {/* 处理进度 */}
            {isProcessing && (
              <div className="bg-white/10 backdrop-blur-md rounded-2xl p-6 border border-white/20">
                <div className="flex items-center space-x-3 mb-4">
                  <Loader2 className="w-6 h-6 text-purple-400 animate-spin" />
                  <span className="text-white font-medium">AI正在处理中...</span>
                </div>
                <div className="w-full bg-black/30 rounded-full h-3 mb-2">
                  <div
                    className="bg-gradient-to-r from-purple-500 to-pink-500 h-3 rounded-full transition-all duration-500"
                    style={{ width: `${progress}%` }}
                  ></div>
                </div>
                <p className="text-sm text-gray-400">进度: {progress}%</p>
              </div>
            )}

            {/* 处理结果 */}
            {processedUrl && (
              <div className="bg-white/10 backdrop-blur-md rounded-2xl p-6 border border-white/20">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-white">处理结果</h3>
                  <div className="flex items-center space-x-2 text-green-400">
                    <Check className="w-5 h-5" />
                    <span className="text-sm">完成 ({processingTime.toFixed(1)}s)</span>
                  </div>
                </div>
                
                <div className="relative aspect-video bg-black/30 rounded-lg overflow-hidden mb-4">
                  {/* 透明网格背景 */}
                  <div className="absolute inset-0 opacity-20"
                    style={{
                      backgroundImage: `url("data:image/svg+xml,%3csvg width='20' height='20' xmlns='http://www.w3.org/2000/svg'%3e%3cdefs%3e%3cpattern id='a' patternUnits='userSpaceOnUse' width='20' height='20'%3e%3crect fill='%23ffffff' width='10' height='10'/%3e%3crect fill='%23f0f0f0' x='10' width='10' height='10'/%3e%3crect fill='%23f0f0f0' y='10' width='10' height='10'/%3e%3crect fill='%23ffffff' x='10' y='10' width='10' height='10'/%3e%3c/pattern%3e%3c/defs%3e%3crect fill='url(%23a)' width='100%25' height='100%25'/%3e%3c/svg%3e")`
                    }}
                  ></div>
                  <img
                    src={processedUrl}
                    alt="处理结果"
                    className="relative w-full h-full object-contain"
                  />
                </div>

                <div className="flex space-x-3">
                  <button
                    onClick={downloadResult}
                    className="flex-1 py-3 bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600 text-white font-semibold rounded-lg transition-all duration-300 transform hover:scale-105"
                  >
                    <div className="flex items-center justify-center space-x-2">
                      <Download className="w-5 h-5" />
                      <span>下载结果</span>
                    </div>
                  </button>
                  <button
                    onClick={reset}
                    className="px-6 py-3 bg-white/10 hover:bg-white/20 text-white font-semibold rounded-lg transition-colors"
                  >
                    处理新图片
                  </button>
                </div>
              </div>
            )}

            {/* 模型信息 */}
            {selectedFile && (
              <div className="bg-white/5 backdrop-blur-md rounded-2xl p-6 border border-white/10">
                <h3 className="text-lg font-semibold text-white mb-4">当前设置</h3>
                <div className="space-y-3 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-400">AI模型:</span>
                    <span className="text-white">{models[settings.model]?.name}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">处理模式:</span>
                    <span className="text-white">{modes[settings.mode]}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">边缘优化:</span>
                    <span className="text-white">{edgeOptions[settings.edgeOptimization]}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">输出格式:</span>
                    <span className="text-white">{outputFormats[settings.outputFormat]}</span>
                  </div>
                </div>
                
                {/* 模型性能指标 */}
                <div className="mt-4 pt-4 border-t border-white/10">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-gray-400">处理速度:</span>
                    <div className="flex space-x-1">
                      {[...Array(5)].map((_, i) => (
                        <div
                          key={i}
                          className={`w-2 h-2 rounded-full ${
                            i < models[settings.model]?.speed 
                              ? 'bg-green-400' 
                              : 'bg-gray-600'
                          }`}
                        ></div>
                      ))}
                    </div>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400">处理质量:</span>
                    <div className="flex space-x-1">
                      {[...Array(5)].map((_, i) => (
                        <div
                          key={i}
                          className={`w-2 h-2 rounded-full ${
                            i < models[settings.model]?.quality 
                              ? 'bg-purple-400' 
                              : 'bg-gray-600'
                          }`}
                        ></div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* 功能介绍 */}
        {!selectedFile && (
          <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="text-center p-6 bg-white/5 backdrop-blur-md rounded-2xl border border-white/10">
              <div className="w-12 h-12 mx-auto mb-4 bg-gradient-to-r from-purple-400 to-pink-400 rounded-full flex items-center justify-center">
                <Zap className="w-6 h-6 text-white" />
              </div>
              <h3 className="text-lg font-semibold text-white mb-2">AI智能识别</h3>
              <p className="text-gray-400">采用最新的深度学习模型，自动识别前景和背景，精确分离</p>
            </div>
            
            <div className="text-center p-6 bg-white/5 backdrop-blur-md rounded-2xl border border-white/10">
              <div className="w-12 h-12 mx-auto mb-4 bg-gradient-to-r from-blue-400 to-cyan-400 rounded-full flex items-center justify-center">
                <ImageIcon className="w-6 h-6 text-white" />
              </div>
              <h3 className="text-lg font-semibold text-white mb-2">多格式支持</h3>
              <p className="text-gray-400">支持JPG、PNG、WebP等多种图片格式，输出高质量透明背景图片</p>
            </div>
            
            <div className="text-center p-6 bg-white/5 backdrop-blur-md rounded-2xl border border-white/10">
              <div className="w-12 h-12 mx-auto mb-4 bg-gradient-to-r from-green-400 to-emerald-400 rounded-full flex items-center justify-center">
                <Settings className="w-6 h-6 text-white" />
              </div>
              <h3 className="text-lg font-semibold text-white mb-2">专业调优</h3>
              <p className="text-gray-400">提供多种处理模式和边缘优化选项，满足不同场景需求</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default App;
