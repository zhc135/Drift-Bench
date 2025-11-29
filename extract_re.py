import os
import json
import glob

def extract_and_summarize_summary_files(folder_path):
    """
    从指定文件夹中提取满足条件的news_n_summary.json文件并汇总
    
    Args:
        folder_path (str): 文件夹路径
    """
    
    # 存储所有满足条件的文件数据
    summary_data_list = []
    
    # 使用glob模式匹配所有news_n_summary.json文件
    pattern = os.path.join(folder_path, "news_*_summary.json")
    summary_files = glob.glob(pattern)
    
    print(f"找到 {len(summary_files)} 个summary文件")
    
    for file_path in summary_files:
        try:
            # 从文件名中提取ID (n)
            filename = os.path.basename(file_path)
            # 提取三位数的n
            file_id = filename.split('_')[1]  # 格式: news_123_summary.json -> 提取123
            
            # 读取JSON文件
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 检查条件
            overall_stats = data.get("overall_stats", {})
            total_versions = overall_stats.get("total_versions", 0)
            average_success_ratio = overall_stats.get("average_success_ratio", 0)
            
            # 条件判断：total_versions >= 3 且 average_success_ratio >= 0.6
            if total_versions >= 3 and average_success_ratio >= 0.6:
                # 提取所需信息
                extracted_data = {
                    "id": file_id,
                    "input_text": data.get("input_text", ""),
                    "rewrite_versions": []
                }
                
                # 提取三个改写版本（去除version_score和quality_label）
                top_rewrites = data.get("top_rewrites", [])
                for rewrite in top_rewrites[:3]:  # 只取前三个版本
                    extracted_rewrite = {
                        "version_id": rewrite.get("version_id", ""),
                        "success_ratio": rewrite.get("success_ratio", 0),
                        "final_rewritten_text": rewrite.get("final_rewritten_text", "")
                    }
                    extracted_data["rewrite_versions"].append(extracted_rewrite)
                
                summary_data_list.append(extracted_data)
                print(f"文件 {filename} 满足条件，已添加到汇总列表")
            else:
                print(f"文件 {filename} 不满足条件 (total_versions: {total_versions}, average_success_ratio: {average_success_ratio})")
                
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")
            continue
    
    # 按照id的数字顺序排序
    summary_data_list.sort(key=lambda x: int(x["id"]))
    
    # 保存汇总结果到原文件夹
    if summary_data_list:
        output_file = os.path.join(folder_path, "summary_collection.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data_list, f, ensure_ascii=False, indent=2)
        
        print(f"\n汇总完成！共处理 {len(summary_data_list)} 个满足条件的文件")
        print(f"结果已保存到: {output_file}")
        
        # 显示排序后的id顺序
        sorted_ids = [item["id"] for item in summary_data_list]
        print(f"按顺序排列的ID: {sorted_ids}")
    else:
        print("\n没有找到满足条件的文件")

def main():
    """主函数"""
    # 自定义文件夹路径
    folder_path = input("请输入文件夹路径: ").strip()
    
    # 检查路径是否存在
    if not os.path.exists(folder_path):
        print("错误：指定的文件夹路径不存在！")
        return
    
    if not os.path.isdir(folder_path):
        print("错误：指定的路径不是一个文件夹！")
        return
    
    # 执行提取和汇总
    extract_and_summarize_summary_files(folder_path)

if __name__ == "__main__":
    main()
