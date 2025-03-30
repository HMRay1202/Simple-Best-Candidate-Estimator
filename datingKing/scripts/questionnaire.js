const { createApp } = Vue

const app = createApp({
    data() {
        return {
            formData: {
                name: '',
                gender: 'male',
                age: 25,
                height: 170,
                education: '',
                school: '',
                occupation: '',
                income: 0,
                appearance: 0,
                personality: 0,
                communication: 0,
                values: 0,
                meetingTimes: 1,
                overallRating: 0,
                result: '',
                notes: ''
            },
            rules: {
                name: [
                    { required: true, message: '请输入姓名', trigger: 'blur' }
                ],
                gender: [
                    { required: true, message: '请选择性别', trigger: 'change' }
                ],
                age: [
                    { required: true, message: '请输入年龄', trigger: 'blur' }
                ],
                education: [
                    { required: true, message: '请选择学历', trigger: 'change' }
                ],
                result: [
                    { required: true, message: '请选择最终结果', trigger: 'change' }
                ]
            },
            savedCandidates: []
        }
    },
    methods: {
        submitForm() {
            this.$refs.candidateForm.validate((valid) => {
                if (valid) {
                    // 深拷贝当前表单数据
                    const candidateData = JSON.parse(JSON.stringify(this.formData))
                    
                    // 添加时间戳
                    candidateData.timestamp = new Date().toISOString()
                    
                    // 保存到本地数组
                    this.savedCandidates.push(candidateData)
                    
                    // 保存到localStorage
                    this.saveToLocalStorage()
                    
                    // 显示成功消息
                    ElementPlus.ElMessage({
                        message: '数据保存成功！',
                        type: 'success'
                    })
                    
                    // 重置表单
                    this.resetForm()
                } else {
                    ElementPlus.ElMessage({
                        message: '请填写必填项！',
                        type: 'warning'
                    })
                    return false
                }
            })
        },
        
        resetForm() {
            this.$refs.candidateForm.resetFields()
        },
        
        deleteCandidate(index) {
            ElementPlus.ElMessageBox.confirm(
                '确定要删除这条记录吗？',
                '警告',
                {
                    confirmButtonText: '确定',
                    cancelButtonText: '取消',
                    type: 'warning',
                }
            ).then(() => {
                this.savedCandidates.splice(index, 1)
                this.saveToLocalStorage()
                ElementPlus.ElMessage({
                    type: 'success',
                    message: '删除成功',
                })
            }).catch(() => {})
        },
        
        exportData() {
            if (this.savedCandidates.length === 0) {
                ElementPlus.ElMessage({
                    message: '没有可导出的数据！',
                    type: 'warning'
                })
                return
            }
            
            const dataStr = JSON.stringify(this.savedCandidates, null, 2)
            const dataBlob = new Blob([dataStr], { type: 'application/json' })
            const url = window.URL.createObjectURL(dataBlob)
            const link = document.createElement('a')
            link.href = url
            link.download = `dating_candidates_${new Date().toISOString().split('T')[0]}.json`
            document.body.appendChild(link)
            link.click()
            document.body.removeChild(link)
            window.URL.revokeObjectURL(url)
            
            ElementPlus.ElMessage({
                message: '数据导出成功！',
                type: 'success'
            })
        },
        
        saveToLocalStorage() {
            localStorage.setItem('datingCandidates', JSON.stringify(this.savedCandidates))
        },
        
        loadFromLocalStorage() {
            const savedData = localStorage.getItem('datingCandidates')
            if (savedData) {
                this.savedCandidates = JSON.parse(savedData)
            }
        }
    },
    mounted() {
        this.loadFromLocalStorage()
    }
})

// 使用Element Plus
app.use(ElementPlus)

// 挂载应用
app.mount('#app') 