const { createApp } = Vue

const app = createApp({
    data() {
        return {
            activeIndex: '/'
        }
    },
    methods: {
        startQuestionnaire() {
            window.location.href = '/questionnaire'
        }
    }
})

// 使用Element Plus
app.use(ElementPlus)

// 挂载应用
app.mount('#app') 