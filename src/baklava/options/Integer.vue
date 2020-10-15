<template>
  <div class="d-sm-flex">
    <div class="ml-1">{{ name }}</div>
    <IntegerInc :index=0
                :value=this.value
                @value-change="updateValue"></IntegerInc>
  </div>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';
import IntegerInc from '@/baklava/input/IntegerInc.vue';

@Component({
  components: { IntegerInc },
})
export default class Integer extends Vue {
  @Prop() value!: number;

  @Prop({ type: String }) name!: string;

  updateValue(value: number) {
    this.$emit('input', value);
  }
}
</script>
