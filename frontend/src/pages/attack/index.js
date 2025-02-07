import { useState, useEffect } from "react";
import { useRouter } from 'next/router';
import Info from '../../components/Info';
import Navbar from "../../components/Navbar";

export default function Home() {
  const router = useRouter();

  const [image, setImage] = useState(null);
  const [imagePreview, setImagePreview] = useState("");
  const [imageUrl, setImageUrl] = useState("");
  const [swappedImage, setSwappedImage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [showInfo, setShowInfo] = useState(false);
  
  useEffect(() => {
    if(!image) return;
    uploadImage();
  }, [image]);

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      setImagePreview(URL.createObjectURL(file)); // Show image preview
    }
  };

  // Upload image and get the image URL
  const uploadImage = async () => {
    if (!image) {
      alert("Please upload an image first.");
      return;
    }

    setIsLoading(true);
    try {
      const formData = new FormData();
      formData.append("image", image);

      const response = await fetch("http://localhost:8000/upload-image", {
        method: "POST",
        body: image,
      });

      if (!response.ok) {
        throw new Error("Failed to upload image.");
      }

      const data = await response.json(); // Get the response data
      setImageUrl(data.image_url); // Set the image URL returned from the server
    } catch (error) {
      console.error("Error uploading image:", error);
      alert("Failed to upload image. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  
  
  // 페이스스왑 요청 핸들러
  const faceswap = async () => {
    if (!image) {
      alert("Please upload an image first!");
      return;
    }

    setIsLoading(true);
    try {
      const formData = new FormData();
      formData.append("image", image); // 서버로 전송할 이미지 추가

      // API 요청
      const response = await fetch("http://localhost:8000/faceswap", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to process the image.");
      }

      const data = await response.json(); // 서버로부터 데이터 받기
      setSwappedImage(data.swappedImageUrl); // 워터마킹된 이미지 URL 저장
    } catch (error) {
      console.error("Error during watermarking:", error);
      alert("Failed to watermark the image. Please try again.");
    } finally {
      setIsLoading(false); // 로딩 종료
    }
  };

  //메세지
  const sendMessage = () => {
    alert('your image is attacked!');
  }

  // 로딩 스피너 컴포넌트
  const LoadingSpinner = () => {
    return (
      <div className="flex items-center justify-center mt-2">
        <div className="w-8 h-8 border-4 border-blue-400 border-t-transparent rounded-full animate-spin"></div>
      </div>
    );
  }

  return (
    <div className="flex items-center justify-center h-screen bg-gray-100">
      {/* Outer Container with Gradient Border */}
      <div className="w-full h-full bg-red-300 p-3">
        {/* Inner Box */}
        <div className="flex h-full bg-neutral-100">
          {/* Navigation Bar */}
          <Navbar />

          {/* Main Content */}
          <div className="flex-1 p-8">
            <h1 className="text-5xl font-bold text-red-500">
              FaceSwap
            </h1>
            <p className="mt-4 text-black font-semibold">
              Upload images to swap faces. Watermarked image will not be swapped well.
            </p>
             {/* Image Upload */}
             <div className="mt-6 w-64 h-64 rounded-lg bg-gray-400">
              {!imagePreview && (
                <div className="w-full h-full rounded-lg">
                  <button
                    className="bg-red-300 text-white w-full rounded-lg h-full shadow text-2xl hover:bg-red-200"
                    onClick={() => document.getElementById("image-upload").click()}
                  >
                    <span className="material-icons text-white text-3xl">add</span>
                  </button>
                  <input
                    id="image-upload"
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={handleImageUpload}
                  />
                </div>
              )}
              {/* Image Preview */}
              {imagePreview && (
                <div className="w-full h-full relative rounded-lg">
                  {/* X 버튼 */}
                  <button
                    className="absolute top-4 right-4 p-2 text-gray-500 rounded-lg hover:text-gray-700 focus:outline-none"
                    onClick={() => setImagePreview(null)}
                  >
                    ✕
                  </button>
                  <img
                    src={imagePreview}
                    alt="Uploaded Preview"
                    className="mt-4 w-full h-full object-cover border-2 border-orange-400"
                  />
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
      <Info
        show={showInfo}
        onClose={() => setShowInfo(false)}
      />
    </div>
  );
}
